"""llm_test.py

Quick single-query test for the NetSoft 2026 LLM-first pipeline.

Run from project root (where distance.csv and city.csv exist):
  python llm_test.py
"""

import json

import llm_agent


def main() -> None:
    # Pick an installed Ollama model
    llm_agent.set_model(llm_agent.get_model())

    query = (
        "Using ONLY the topology snapshot provided, compute the shortest path between M1 and M3 "
        "by total distance_km and report hop_count and total_distance_km. "
        "Return exactly one JSON object."
    )

    out = llm_agent.run(query, mode="none", rag_enabled=False)

    print("OK:", out.get("ok"))
    print("Prompt tokens:", out.get("prompt_tokens"), "Completion tokens:", out.get("completion_tokens"), "Total:", out.get("total_tokens"))

    if out.get("ok"):
        print(json.dumps(out.get("llm_json"), indent=2))
    else:
        print("RAW OUTPUT:\n", out.get("raw"))


if __name__ == "__main__":
    main()
