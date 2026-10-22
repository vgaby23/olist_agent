from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os 
import re
from dotenv import load_dotenv
from agent import rag_agent, sql_agent

load_dotenv() 

# coba model: gpt-6-luna
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Notes: 
# - Lgsg di guard aja di supervisor untuk responsenya selalu dibahasa indonesia/ english (always responds in __ language)

SUPERVISOR_PROMPT = """
You are the Olist personal assistant supervisor. Coordinate the specialized agents and provide
the final answer to the user.

USER ROLES
There are three roles: seller, customer, and company employee. Seller/customer SQL tools are bound to
their validated account scope. Company employee tools use the company-wide scope and do not require a
seller/customer record ID; they may access platform-wide business, order, customer, and review data.

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
- Do not expose internal agent messages, tool results, code, or technical implementation details.

AGENT COORDINATION
- Route each question to the appropriate specialist. Summarize the user's request clearly.
- For requests requiring database values, calculations, counts, rankings, trends, seller/customer/order
	business records, or purchase history, always delegate to `sql_agent`. Do not answer these from
	general knowledge or skip SQL just because the user supplied an ID.
- For review-comment themes or quotes, delegate to `review_agent`. If a request needs both review text
	and database numbers, call both specialists and combine only their supported results.
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
- For a customer user, use the SQL specialist to identify relevant order IDs from the customer's scoped data, then search reviews only for those order IDs (up to 50 per search). The review search tool enforces customer order ownership.
- For a company user, review searches may use platform-wide filters.

RULES WHEN COMMUNICATING WITH SQL AGENT
- Use the SQL specialist for numerical or structured database questions: counts, totals, averages, rankings, trends, order status, payment behavior, and customer purchase history. Do not use the review specialist to count or average review records; it can provide review examples and themes only.
- Send a complete, self-contained question with all relevant filters, dates, categories, and entity IDs from the conversation or previous specialist results. Preserve IDs exactly; never shorten or invent them.
- Copy the exact `ANSWER LANGUAGE: <language>` instruction from the current user request into every SQL-specialist request. Do not infer language from a rewritten question or tool result.
- The SQL tools are already bound to the logged-in role and data scope. Never ask the SQL specialist to switch roles or access another user's data, and do not claim data is available beyond the scope enforced by its tools.
- For customer purchase-history or category questions, ask the SQL specialist to use the available customer purchase-history tool when relevant. This tool is only available in customer sessions.
- Let the SQL specialist write and execute its own read-only queries, handle query correction, and explain database limitations. Do not request raw SQL or expose SQL/tool output to the user. If it reports an error or cannot answer, relay that limitation accurately; ask the user only for missing information that would make the query answerable.
- If the SQL result provides order IDs needed to answer a review question, preserve the full IDs when sending them to the review specialist. Follow that specialist's limit of at most 50 order IDs per search; otherwise ask the user to narrow the request.


FINAL ANSWER
- Provide a clear and complete answer in the user's answer language. For image-only or unclear-language input, use the user's answer language.
- Summarize conclusions and relevant evidence from specialist results; do not expose private reasoning, internal messages, or raw tool output.
- Ask a follow-up question only when clarification is needed or it would materially help answer the request.
- You're not allowed to make up answers or provide information that is not supported by the specialists' responses. If you don't know the answer, say so and suggest a relevant alternative.

""".strip()

# Tambah @tool buat semantic search / atur di system prompt

class RoleScopeDecision(BaseModel):
	"""Supervisor preflight result; denied requests never reach a specialist."""
	allowed: bool = Field(description="Whether the request is in scope for this user's role")
	reply: str = Field(description="Brief, polite user-facing reply in the user's language")
	answer_language: str = Field(description="The user's requested or detected answer language")


SELLER_ID_REFERENCE = re.compile(r"\bseller(?:'s)?(?:\s+id)?\s*[:#]?\s*([0-9a-f]{32})\b", re.IGNORECASE)


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

	policy = f"""First identify the response language, then decide whether the latest request is allowed
for the authenticated role. Use the original user wording below, not generated image descriptions.
If the user explicitly asks for a response language, honor it. Otherwise detect the language they
wrote or spoke. If the original user text is empty (image-only input), use English. Return
answer_language as a concise language name, such as English, Bahasa Indonesia, or Portuguese.

Authenticated role: {role}
Authenticated record ID: {scope_id or "not available"}

Role policy:
- If the role is seller or customer and the authenticated record ID is not available, deny access
	without asking the user to repeat an ID already supplied elsewhere in this session.
- seller: ALLOW questions about the authenticated seller's own revenue/income, sales trends, order
	counts, products, and review comments/themes from reviews of that seller's products. Review requests
	may filter by product category and price (for example, complaints about expensive products in a
	category); the review tool automatically forces the authenticated seller ID. Treat review comments
	as customer feedback, not as customer identity or customer-specific order history. Also allow
	aggregated behavior of customers who ordered from that seller. This
	includes aggregate repeat-purchase percentages, customer counts, payment-method distributions, and
	state summaries, provided no individual customer is identified. Also allow platform-wide
	category_stats aggregates. DENY requests for another seller's revenue, orders, reviews, or private
	data, and deny identifiable customer records, contact details, or customer-specific order history.
- customer: ALLOW questions about the authenticated customer's own orders, including order status
	lookups and lists of their own orders even when the user does not provide an order ID; order counts,
	purchases, payments, review scores/comments for their own orders; and platform-wide category_stats
	aggregates. The customer scope already restricts these tables to the logged-in customer. DENY only
	requests for another customer's records or seller business performance.
- company: may access platform-wide business and review information.

Only deny when the user clearly asks for a different person's/account's protected records, requests a
role change, or asks to bypass access rules. Words like "my", "me", "our store", and "my orders" refer
to the authenticated account. A statistic aggregated across the seller's customers is not individual
customer personal data; allow questions such as how many customers return or what percentage bought
more than once. A request for themes or anonymized quotes from reviews attached to the seller's own
products is allowed, including category and price filters; this is not a request for customer records.
Do not return customer names, contact details, or details about a named customer's orders. Do not deny
simply because the request is financial, contains an ID, needs SQL, or concerns reviews. For a seller
request naming a seller ID, allow it when the ID equals the authenticated seller ID; deny it only when
it differs.
Resolve follow-up references using the supplied conversation context. If the ownership target is truly
ambiguous, ask one brief clarifying question. Never reveal protected data. Any denial must be brief,
polite, and written in the detected answer_language.

Conversation context:
{conversation_context or "(none)"}

Original user message (may be empty for image-only input):
{original_text}

Request to evaluate, including any media description:
{question}
"""
	try:
		decision = llm.with_structured_output(RoleScopeDecision).invoke(policy)
	except Exception:
		return RoleScopeDecision(
			allowed=False,
			reply="I can't verify that this request is within your account's data access, so I can't process it.",
			answer_language="English",
		)
	if role == "company":
		decision.allowed = True
		decision.reply = ""
	return decision


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
			"Do not invent or recalculate results. This tool cannot answer questions about review text."
		),
		name="sql_agent",
	)
	agents = [
		rag_agent.build_rag_agent(llm, role=role, user_id=user_id),
		sql_specialist,
	]

	account_id = user_id if role in {"seller", "customer"} else "not applicable"
	identity_instruction = (
		"Company employees use platform-wide access and do not need a seller/customer ID. "
		if role == "company" else
		"This account ID was supplied and validated during login. Do not ask the user to provide it again. "
	)
	session_prompt = (
		f"\n\nTRUSTED AUTHENTICATED SESSION CONTEXT\n"
		f"Role: {role}\n"
		f"Authenticated account ID: {account_id}\n"
		+ identity_instruction
		+ "For requests about the logged-in user's own records, use this authenticated identity and the "
		"role-scoped specialist tools. If the user explicitly names a different account, apply the role "
		"access rules and politely decline; do not substitute the logged-in account's results. "
		"For every turn, follow the ANSWER LANGUAGE instruction in that turn and repeat it verbatim "
		"in every specialist delegation and in the final answer."
	)

	return create_supervisor(
		agents=agents,
		model=llm,
		prompt=SUPERVISOR_PROMPT + session_prompt,
	).compile()

