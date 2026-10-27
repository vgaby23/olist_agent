
def activity_label(event):
    """Map graph node events to safe, user-facing progress labels."""
    namespace = "/".join(event.get("ns", ()))
    if "review_agent" in namespace:
        return "Review specialist is searching customer feedback"
    if "sql_agent" in namespace or "data_agent" in namespace:
        return "SQL specialist is analyzing Olist data"

    if event.get("type") == "updates":
        node_names = event.get("data", {}).keys()
        if "supervisor" in node_names:
            return "The agent is preparing an answer"
        if any(name.startswith("transfer_to_") for name in node_names):
            return "The agent is handing the request to a specialist"
    return None