from typing import Annotated

from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
import os 
import re
from dotenv import load_dotenv
from agent import category_resolver, policy, rag_agent, sql_agent
from agent.llm import chat_model

load_dotenv()

# coba model: gpt-6-luna (chat_model tidak mengirim temperature untuk model yang menolaknya)
llm = chat_model(temperature=0)

# Notes: 
# - Lgsg di guard aja di supervisor untuk responsenya selalu dibahasa indonesia/ english (always responds in __ language)

SUPERVISOR_PROMPT = """
You are the Olist personal assistant supervisor. Coordinate the specialized agents and provide
the final answer to the user.

USER ROLES
There are three roles: seller, customer, and company employee. Seller tools are bound to the validated
seller ID. A customer is an anonymous shopper (no ID, like a general chatbot) and may only get public,
platform-wide product and review information. Company employee tools use the company-wide scope; they may
access platform-wide business, order, customer, and review data. The allowed and forbidden actions of the
logged-in role are listed in the session context below; the specialist tools also enforce them in code.

PRODUCT TYPES AND CATEGORIES
- The data only knows product CATEGORIES (exact English names such as health_beauty); products have no names.
- When the user names a product type, product or brand instead of an exact category ("make up", "lipstik",
  "sepatu lari", "Maybelline"), call `find_product_category` with that wording FIRST, then put the exact
  category name it returns into every specialist request.
- If it returns two close candidates, use both when that answers the question naturally, otherwise ask the
  user which one they mean. If it returns no match, tell the user no matching category exists; never guess.

SPECIALISTS
- SQL specialist: numerical questions about sales, customers, products, ratings, and business data
- Review specialist: customer review comments, complaints, praise, and product feedback.

LANGUAGE REQUIREMENTS
- Determine the answer language from the user's original typed message or audio transcription before delegating. If the user explicitly requests a response language, honor that request; otherwise use the language the user wrote or spoke.
- Respond entirely in that language. Do not switch language based on a supervisor rewrite, image description, specialist response, SQL result, or Portuguese review quote.
- Include an explicit `ANSWER LANGUAGE: <language>` instruction in every specialist request. Require every specialist to answer in that language.
- For image-only requests without text or audio, use English as the fallback response language.
- For review answers, write the summary and displayed quotes in the answer language. If the answer
	language is not Portuguese, do not show Portuguese source quotes or bilingual quote text. Preserve
	the exact rating, category, city, and order ID metadata returned by the review specialist.

USER INTERACTION
- Be polite, professional, clear, and concise.
- Greet the user on first contact when appropriate; do not repeat the greeting on every turn.
- If the user attaches only an image without context, ask what they want to know about it.
- If a request is unrelated to Olist, briefly explain that you can help only with Olist-related topics.
- If a request is vague, ask for clarification before making assumptions.
- If the requested item by the user is not specific on one item, you should asked the user again before proceeding. 
- Do not expose internal agent messages, tool results, code, or technical implementation details (the
	database schema is not one of these for a company user; see the session context).

AGENT COORDINATION
- Route each question to the appropriate specialist. Summarize the user's request clearly.
- For requests requiring database values, calculations, counts, rankings, trends, seller/customer/order
	business records, or purchase history, always delegate to `sql_agent`. Do not answer these from
	general knowledge or skip SQL just because the user supplied an ID.
- For review-comment themes or quotes, delegate to `review_agent`. If a request needs both review text
	and database numbers, call both specialists and combine only their supported results.
- Every transfer takes a `task`: the specialist sees ONLY that task, not the conversation. Write it
	complete and self-contained (IDs, categories, ratings, period, filters, ANSWER LANGUAGE) and give each
	specialist only the part it can answer.
- The database has review SCORES but no review TEXT. Never ask `sql_agent` for review text, complaint
	reasons or themes; send only the numerical part to `sql_agent`, then transfer to `review_agent` for the
	text part. If `sql_agent` says it has no review text, that is expected: go to `review_agent` next.
- The review search can filter by product category, product ID, rating, price and SELLER city, but NOT by
	customer state/city, purchase month, payment type or delivery time. For reviews of such a group (for
	example "what do customers in Roraima say"), first ask `sql_agent` for up to 50 order IDs of delivered
	orders in that group (prefer orders that have a review, newest first), then ask `review_agent` to search
	those `order_ids` in ONE search. For several groups (several states), ask for up to 10 order IDs per group.
- For requests asking for products with bad reviews plus a sales quantity, coordinate both specialists
	in this order: (1) ask `sql_agent` for the five product IDs with the most delivered low-rated reviews
	(scores 1-2), including each product's units sold in the applicable period; (2) for each returned
	product ID, ask `review_agent` for up to two actual low-rated review examples, using that product ID
	and `max_rating=2`; (3) present every product returned by SQL, pairing the exact SQL product ID and
	units-sold count with the quotes. If RAG has no matching quote for one product, keep the product and
	state that no quote was retrieved. Never substitute a quote from another product or invent names/text.
- Interpret “bad/poor review” as review score 1 or 2 unless the user specifies another threshold.
	Interpret “nominal terjual / units sold” as count of delivered order-item rows in the same period.
- The SQL specialist exposes the `ask_data_agent` tool. Require it to call that tool for SQL-backed
	questions and return its result; it is not a direct database tool on the supervisor.
- The SQL specialist is available only when this session has a valid data scope. If it is not
	available, do not invent numerical answers; explain that the data access is not configured.
- If a specialist result is incomplete or a tool reports a recoverable error, use the specialist's supported retry behavior. Ask the user a clarifying question only when required information is missing or the request is genuinely ambiguous.
- If agents disagree, assess their evidence and give the most accurate supported answer.
- If no available specialist can answer, explain that in the user's language and suggest a relevant alternative.
- When the user asked for review content and a numerical result, consult both specialists without asking for confirmation. If the requested combination is unclear, ask a focused clarifying question.

RULES WHEN COMMUNICATING WITH RAG AGENT
- Send a COMPLETE question. The RAG does not see the conversation, so write the category, rating or city into the question itself.
- Say in the question which LANGUAGE the user wrote in (e.g. "Answer in Bahasa Indonesia"). For
non-Portuguese answers, show only translated review text; do not expose the Portuguese source quote.
For Portuguese answers, the original quote may be shown. Without a hint, the agent uses the language
of the question you send it.
- To get the review of ONE order, put the order_id in the question: "What did order <order_id> say?". Price limits ("above 150 BRL") also work as filters.
- For a SELLER user, put their seller_id in the question: "Why do customers of seller <seller_id> give
low ratings?". The same works for one product: "What do reviews of product <product_id> say?".
- For a seller user, always scope review requests to the authenticated seller ID. The review search tool enforces this scope and rejects a different seller ID.
- For a customer user, ask for reviews by product category, product type, rating, price range or seller city only. Never send a seller ID or order ID; the review search tool rejects them. Do not show order IDs to a customer.
- For a company user, review searches may use platform-wide filters.

RULES WHEN COMMUNICATING WITH SQL AGENT
- Use the SQL specialist for numerical or structured database questions: counts, totals, averages, rankings, trends, order status, payment behavior, and customer purchase history. Do not use the review specialist to count or average review records; it can provide review examples and themes only.
- Send a complete, self-contained question with all relevant filters, dates, categories, and entity IDs from the conversation or previous specialist results. Preserve IDs exactly; never shorten or invent them.
- Copy the exact `ANSWER LANGUAGE: <language>` instruction from the current user request into every SQL-specialist request. Do not infer language from a rewritten question or tool result.
- The SQL tools are already bound to the logged-in role and data scope. Never ask the SQL specialist to switch roles or access another user's data, and do not claim data is available beyond the scope enforced by its tools.
- For a customer user, the SQL specialist only has platform-wide aggregates (per category, per product, per seller and category, delivery time per state). Use them for product and seller recommendations (e.g. the best-rated sellers for make up). Do not ask it about specific orders, customers, sellers' revenue or company revenue.
- Let the SQL specialist write and execute its own read-only queries, handle query correction, and explain database limitations. Do not request raw SQL or expose SQL/tool output to the user. If it reports an error or cannot answer, relay that limitation accurately; ask the user only for missing information that would make the query answerable.
- If the SQL result provides order IDs needed to answer a review question, preserve the full IDs when sending them to the review specialist. Follow that specialist's limit of at most 50 order IDs per search; otherwise ask the user to narrow the request.


FINAL ANSWER
- Provide a clear and complete answer in the user's answer language. For image-only or unclear-language input, use the user's answer language.
- Plural ranking questions ("which categories have the best rating", "top products") ask for a ranked list
	(10 items unless the user gives a number); never narrow them to the single top item.
- Only when the user asks for ONE item ("the top one", "my best-selling product") and the SQL specialist
	reports a tie, say so and keep every tied item; a follow-up review search then covers each (at most 3).
- Never write the "ANSWER LANGUAGE: ..." instruction itself in the final answer; it is for the specialists only.
- Summarize conclusions and relevant evidence from specialist results; do not expose private reasoning, internal messages, or raw tool output.
- Keep the period and definition the SQL specialist states with its numbers (for example "January 2017 to
	August 2018", "delivered orders, excluding freight"); never drop them when summarizing.
- Ask a follow-up question only when clarification is needed or it would materially help answer the request.
- You're not allowed to make up answers or provide information that is not supported by the specialists' responses. If you don't know the answer, say so and suggest a relevant alternative.

""".strip()

# Tambah @tool buat semantic search / atur di system prompt

class RoleScopeDecision(BaseModel):
	"""Supervisor preflight result; denied requests never reach a specialist.
	Field order matters: the LLM fills them in this order, so the language is fixed BEFORE the reply is written
	(with reply first, denials often came back in Spanish)."""
	answer_language: str = Field(description="The user's requested or detected answer language")
	allowed: bool = Field(description="Whether the request is in scope for this user's role")
	violated_rule: str = Field(default="", description="Id of the forbidden rule the request breaks (e.g. C1), "
	                           "empty when allowed")
	reply: str = Field(description="Brief, polite user-facing reply written in answer_language")


SELLER_ID_REFERENCE = re.compile(r"\bseller(?:'s)?(?:\s+id)?\s*[:#]?\s*([0-9a-f]{32})\b", re.IGNORECASE)


def id_facts(role, scope_id, text):
	"""Compare seller ids in the request with the logged-in seller IN CODE (the LLM was unreliable at it)."""
	if role != "seller" or not scope_id:
		return ""
	mentioned = {match.lower() for match in SELLER_ID_REFERENCE.findall(text or "")}
	if not mentioned:
		return ""
	own = sorted(mentioned & {scope_id})
	other = sorted(mentioned - {scope_id})
	facts = []
	if own:
		facts.append(f"The seller ID {own[0]} in the request IS the authenticated seller's own ID (allowed).")
	if other:
		facts.append(f"The seller ID(s) {', '.join(other)} in the request belong to ANOTHER seller: only their "
		             "public rating, review volume and sales rank are allowed, anything else is S1.")
	return "\nChecked in code: " + " ".join(facts) + "\n"


def check_request_scope(role, question, user_id=None, conversation_context="", original_text=""):
	"""Determine answer language and reject out-of-scope requests before specialist delegation."""
	role = (role or "").strip().lower()
	if role not in {"company", "seller", "customer"}:
		return RoleScopeDecision(
			allowed=False,
			reply="I can't help with this request because the account role is not recognized.",
			answer_language="English",
		)

	scope_id = (user_id or "").strip().lower()

	instructions = f"""First identify the response language, then decide whether the latest request is allowed
for the authenticated role. Use the original user wording below, not generated image descriptions.
If the user explicitly asks for a response language, honor it. Otherwise detect the language of the
ORIGINAL USER MESSAGE below only (the latest turn). Never take the language from the conversation context:
a user may switch language between turns, and an English message gets an English answer even after
Indonesian turns. If the original user text is empty (image-only input), use English. Return
answer_language as a concise language name, such as English, Bahasa Indonesia, or Portuguese.
The reply must be written in answer_language, never in another language.

Authenticated role: {role}
Authenticated seller ID: {scope_id or "not applicable"}

Role policy:
{policy.rules_text(role)}

How to decide:
- If the role is seller and the authenticated seller ID is not available, deny access.
- Deny only when the request clearly matches a FORBIDDEN rule; set violated_rule to that rule's id (for
	example C1). Otherwise allow and leave violated_rule empty.
- seller: words like "my", "me", "our store" refer to the authenticated seller. The seller's OWN revenue,
	income, sales, orders, products, ratings and reviews are always allowed (that is never S4; S4 is only
	about the whole platform). A seller request naming a seller ID is allowed when the ID equals the
	authenticated seller ID and denied (S1) when it differs.{id_facts(role, scope_id, question)}
	Statistics aggregated across the seller's customers (repeat-purchase rate, customer counts, payment
	mix, state summaries) are not individual customer data; allow them. Themes or quotes from reviews of
	the seller's own orders are allowed, including category and price filters. The public aggregates a
	shopper sees (other sellers' rating, review volume and sales rank, best-rated sellers per category) are
	allowed too; only other sellers' review comments, revenue, exact sales or orders are S1/S2.
- customer: the user is anonymous, so "my order" or "my purchases" cannot be looked up; deny those with
	C2 or C3 and explain that order lookups are not available in this chat. Questions about a product
	category, product type, brand or price range, and what reviewers say about it, are allowed even when
	they mention bad sellers in general ("are sellers of shoes slow?"). Recommending or ranking sellers by
	rating or sales rank ("5 sellers with the best reviews for make up") is allowed, and so is a seller's
	public rating; only the review comments or revenue/sales figures of a SPECIFIC seller are C1.
- Do not deny simply because the request contains an ID, needs SQL, or concerns reviews.
Use the conversation context only to resolve follow-up references, never to choose the language. If the target is truly
ambiguous, ask one brief clarifying question. Never reveal protected data. Any denial must be brief,
polite, say what the user CAN ask instead, and be written in the detected answer_language.

Conversation context:
{conversation_context or "(none)"}

Original user message (may be empty for image-only input):
{original_text}

Request to evaluate, including any media description:
{question}
"""
	try:
		decision = llm.with_structured_output(RoleScopeDecision).invoke(instructions)
	except Exception:
		if role == "company":    # company may ask anything, so a failed check must not refuse it
			return RoleScopeDecision(allowed=True, reply="", answer_language="the language of the user's message")
		return RoleScopeDecision(
			allowed=False,
			reply="I can't verify that this request is within your account's data access, so I can't process it.",
			answer_language="English",
		)
	if role == "company":
		decision.allowed = True
		decision.reply = ""
	# S1 = ANOTHER seller. When code proved every seller id in the request is the seller's own, S1 is wrong.
	mentioned = {match.lower() for match in SELLER_ID_REFERENCE.findall(question or "")}
	if role == "seller" and decision.violated_rule == "S1" and mentioned and mentioned <= {scope_id}:
		decision.allowed = True
		decision.reply = ""
	if decision.allowed:
		decision.violated_rule = ""
	return decision


ANSWER_LANGUAGE_LINE = re.compile(r"^\s*\**ANSWER LANGUAGE:[^\n]*\n*", re.IGNORECASE | re.MULTILINE)


def clean_answer(text):
	"""Remove an echoed 'ANSWER LANGUAGE: ...' instruction line from the final answer before it is shown."""
	return ANSWER_LANGUAGE_LINE.sub("", text or "").strip()


def make_handoff_tool(agent_name, description):
	"""transfer_to_<agent_name>(task): the specialist receives ONLY `task`, not the whole conversation.
	With the default handoff (no arguments) the specialist read the full chat and tried to answer every part of
	it, e.g. the SQL specialist kept asking its data tool for review text it does not have (company Q8)."""
	name = f"transfer_to_{agent_name}"

	@tool(name, description=description)
	def handoff(
		task: Annotated[str, "Complete, self-contained request for this specialist only: the part it can answer, "
		                     "with every ID, category, rating, period and filter written out, ending with "
		                     "'ANSWER LANGUAGE: <language>'. The specialist does not see the conversation."],
		state: Annotated[dict, InjectedState],
		tool_call_id: Annotated[str, InjectedToolCallId],
	) -> Command:
		tool_message = ToolMessage(content=f"Successfully transferred to {agent_name}", name=name,
		                           tool_call_id=tool_call_id,
		                           response_metadata={METADATA_KEY_HANDOFF_DESTINATION: agent_name})
		# Like the library's handoff: the call stays in the history. task_only() below hands the specialist the task.
		return Command(goto=agent_name, graph=Command.PARENT,
		               update={"messages": state["messages"] + [tool_message]})

	handoff.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
	return handoff


def task_only(agent):
	"""Wrap a specialist so it receives only the `task` of the latest transfer_to_<name> call, not the chat.
	(Sending the task with Send() instead lost the handoff from the supervisor's history, and it looped.)"""
	call_name = f"transfer_to_{agent.name}"

	def run(state, config):
		task = next((call["args"].get("task", "") for message in reversed(state["messages"])
		             for call in getattr(message, "tool_calls", None) or [] if call["name"] == call_name), "")
		output = agent.invoke({"messages": [HumanMessage(content=task or state["messages"][0].text)]}, config)
		return {"messages": output["messages"][-1:]}

	graph = StateGraph(MessagesState)
	graph.add_node(agent.name, run)
	graph.add_edge(START, agent.name)
	return graph.compile(name=agent.name)


def supervisor_agent(role, user_id=None):
	"""Create a supervisor with specialists scoped to the authenticated user."""
	role = (role or "").strip().lower()
	if role not in {"company", "seller", "customer"}:
		raise ValueError(f"Unsupported user role: {role}")

	sql_specialist = create_agent(
		model=llm,
		tools=sql_agent.make_data_tools(role, user_id),
		system_prompt=(
			"You are the Olist SQL specialist. For questions requiring database facts or numbers, "
			"preserve the exact ANSWER LANGUAGE instruction when calling ask_data_agent. "
			"Return the tool's answer faithfully in that language. "
			"Do not invent or recalculate results. The database has review scores but no review text: "
			"if the request also asks for review text, complaint reasons or themes, answer only the "
			"numerical part, never call ask_data_agent again for the text, and return so the "
			"supervisor can ask the review specialist. If one part of a request is impossible (for example "
			"deleting or changing data, the database is read-only), say so for that part and still call "
			"ask_data_agent for the remaining parts."
		),
		name="sql_agent",
	)
	agents = [
		task_only(rag_agent.build_rag_agent(llm, role=role, user_id=user_id)),
		task_only(sql_specialist),
	]

	account_id = user_id if role == "seller" else "not applicable"
	identity_instruction = {
		"company": "Company employees use platform-wide access and do not need an ID. They may also see the "
		           "database schema (tables, columns, views): when asked, transfer to sql_agent with a task to "
		           "describe the schema, and do not refuse it as an implementation detail. ",
		"seller": "This seller ID was supplied and validated during login. Do not ask the user to provide it again. ",
		"customer": "This is an anonymous shopper without an account. Never ask for or use a customer, order or "
		            "seller ID. ",
	}[role]
	session_prompt = (
		f"\n\nTRUSTED AUTHENTICATED SESSION CONTEXT\n"
		f"Role: {role}\n"
		f"Authenticated account ID: {account_id}\n"
		+ identity_instruction
		+ "Use the role-scoped specialist tools. If the user asks for something FORBIDDEN below, politely "
		"decline and say what they can ask instead; do not substitute other results. "
		"For every turn, follow the ANSWER LANGUAGE instruction in that turn and repeat it verbatim "
		"in every specialist delegation and in the final answer.\n\n"
		+ policy.rules_text(role)
	)

	return create_supervisor(
		agents=agents,
		model=llm,
		tools=[
			make_handoff_tool("sql_agent", "Ask the SQL specialist for numbers from the database (counts, totals, "
			                  "averages, rankings, trends, IDs). It has review SCORES but no review text."),
			make_handoff_tool("review_agent", "Ask the review specialist for what reviews say (themes, complaints, "
			                  "quotes), filtered by category, product ID, rating, price, seller city or order_ids."),
			category_resolver.find_product_category,
		],
		prompt=SUPERVISOR_PROMPT + session_prompt,
	).compile()

