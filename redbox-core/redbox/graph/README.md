# Redbox and LangGraph

The core functionality of Redbox is fundamentally a graph. The user submits a request, and we run steps based on that request until we've finished our result. It's a bit like a factory production line: request in, sequential operations, result out.

Graphs get complicated. We therefore present the following constraints for developing the graph, which we believe will mean the graph never gets too much more complicated to understand than it already is.

> [!IMPORTANT]
> [Low-Level LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/low_level/) is essential reading for developing this part of Redbox.

## Design conventions

Draw your change before you do it in code.

## Node conventions

The graph of Redbox contains three types of node: robots doing jobs on the production line.

* Processes, which affect the state
* Decisions, which have conditional out-edges
* Sends, which have conditional send out-edges

Naming conventions should be honoured:

* `p_`rocesses should be named for what they do: `p_summarise`, `p_retrieve_documents`
* `d_`ecisions should be named for what they decide: `d_docs_bigger_than_context`
* `s_`ends work best with minimal labels: `s_chunk`, `s_group`

Grouping them together in code has also proved helpful to comprehension.

We often use constructor functions. The naming convention is `build_*` followed by the type: `build_*_send`.

### Patterns

A common constructor function is the **pattern**. A pattern constructs processes that affect the state in a single way. For example, the chat pattern looks at `state["request"]` and modifies `state["text"]`.

> [!IMPORTANT]
> Patterns form the atomic process types. By combining them we can make all sorts of Redboxes!

## Edge conventions

Note that for the graph to plot neatly, `path_map` must be set on conditional edges, even for send nodes.

## Test conventions

Unit tests happen at the pattern, subgraph, and graph levels.
