---                                                                                                                                                                                                                                    
layout: post
title: "A Primer on the Underpinnings of LLM Agents"
date: 2024-12-12 7:00
author: pwais
---

_Forward_

This post was written as partial fulfillment of the requirements for the 
[Fall 2024 UC Berkeley LLM Agents MOOC](https://llmagents-learning.org/f24) 
(also known as [CS294/194-196](https://rdi.berkeley.edu/llm-agents/f24)).

_Why_ has interest in  Large Language Models (LLMs) and LLM Agents exploded in the past few years?
_What_ is the underlying research that has changed the technological landscape?
The aforementioned course focuses on LLMs and Agents without providing
much background on machine learning or LLM particulars.  This post seeks to 
provide (briefly) that [missing connection to machine learning](#illya-30-papers)
and to highlight the key research results that supplant today's LLM Agents.

## LLM Foundations

### What is an LLM?

The rise of LLM Agents can be seen as a consequence of seminal progress in the
construction of Large Language Models (LLMs).  Like any trained
[supervised learning model](https://en.wikipedia.org/wiki/Supervised_learning),
an LLM seeks to approximate (or "learn") the behavior of a *function* `y = f(x)` from a
training set of exemplar input-output pairs `(x, y)`.  Most LLMs today, including the 
massively influential OpenAI GPT models such as [GPT-2](https://github.com/openai/gpt-2),
seek to emulate a "next-token-prediction" function (i.e. "next-word-prediction,"
where most words constitute 1-2 tokens).  For example, given an input
string such as `"the cat jumped over the "`, the LLM would seek to learn that
the next word (or token) in the string should be `"dog"`.

Under the hood, most implementations of LLMs today use neural
networks based on the ["Transformer Architecture."](https://arxiv.org/abs/1706.03762)
These neural networks are powerful sequence-to-sequence learners--- where the
input-output `(x, y)` pairs in the training set are full sentences, 
sequences of pixels (i.e. images), or time-ordered waveform datums (i.e. audio
data).  Since neural networks take numeric data as input, LLMs must first
[_tokenize_](https://www.youtube.com/watch?v=zduSFxRajkE) text input into
a sequence of well-defined numbers.  For example, the word `"hello"` might
be the token number `24912` (according to the popular [`tiktokenizer`](https://tiktokenizer.vercel.app/)
library), and the string `"hi cat"` might be the numeric sequence of tokens `[3686, 9059]`.
A tokenizer itself is a bijective mapping between strings and numbers.  Thus an LLM model 
will "encode" string input to numbers, learn / predict sequences
of numbers, and then "decode" those numbers to produce a final output string.

Most LLMs also effectively implement (as part of the Transformer Architecture) a 
_probabilistic model_ of words (or tokens).  Thus for any given input string, such
as `"the cat jumped over the "`, the LLM will actually infer a _probability distribution_
over possible next-word completions.  For example, the word `"dog"` might have
the highest probability (e.g. 51%), but the word `"mouse"` might also have
substantial probability (e.g. 10%).  To generate a complete output (i.e a 
full sentence), most LLMs will _generate_ (or "sample") a single token at a
time based upon all prior tokens (i.e. both the input text and partial output
as the LLM generates the string).  Note that since each token is a "sample" from
a probability distribution, an LLM can generate more than one distinct output
for any given input (and such outputs can be ranked by total probability).

The LLM Agents that we discuss below tend to use LLMs trained on massive 
numbers of strings containing _question-answer pairs_, where a question could be as simple as "what
is 1+1?" and the answer string might be "the answer is 2".  While these training sets
typically include [trillions of tokens](#45tb) of [arbitrary text from the internet](https://en.wikipedia.org/wiki/The_Pile_(dataset))
to support basic language understanding, they also include
specialized text such as [corpora of grade school math word problems](https://github.com/openai/grade-school-math).

_To further explore the underlying details of LLM model implementations, the reader
should consult [the video tutorial of Andrej Karpathy's re-implementation of GPT-2 from scratch](https://www.youtube.com/watch?v=l8pRSuU81PU) as well as the related [Nano GPT code](https://github.com/karpathy/nanoGPT)._


### How can LLMs Perform Reasoning?

A key phenomena leading to the success of LLM Agents is the ability of LLMs
to perform well on question-answer tasks that emulate human _reasoning_.  To
elicit such phenomena, good question-answer pairs (i.e. training data) or questions
(i.e. user prompts at inference time) should exemplify _["chain-of-thought"](https://arxiv.org/pdf/2201.11903)
reasoning_: one or more sentences where the speaker "shows her work."  See a key example below:

<center>
<img 
  alt="Chain-of-Thought prompting from Wei et all" 
  src="{{site.baseurl}}/assets/images/cot_prompting_wei_et_all2.jpg"
  width="550px" style="border-radius: 8px; border: 8px solid #ddd;" />
</center>

This technique of "prompting" or formulating inputs (and/or labeled outputs) 
serves a fundamental basis for LLM agents to:
  * accurately perform simple arithmetic and logical synthesis (with relatively high accuracy);
  * break a larger problem (which might be summarized in a single sentence) into smaller, detailed components (where each part or step might be several sentences);
  * use "tools" such as arbitrary programmatic functions (e.g. `python` code) using textual descriptions of the APIs of the functions as well as _demonstrations_ of the functional behaviors.

In particular, "chain-of-thought" text can be provided not just as training data,
but also solely as part of the user prompt (i.e. at inference time) to help
elucidate desired reasoning that might not appear in the training data.  For
example, a user prompt might include text demonstrating uses of a programmed
function written well after the LLM model was trained.  Thus the "chain-of-thought"
technique unlocks substantial extensibility of trained LLMs (including closed-source,
proprietary LLMs such as OpenAI's GPT).

The initial research on "chain-of-thought" reasoning has inspired a variety of
subsequent study, including the observations that:
 * for more complicated reasoning tasks, [premises must be provided in logical order](https://arxiv.org/abs/2402.08939) 
    (i.e. as a human would expect) for LLMs to succeed;
 * breaking down larger tasks using ["least-to-most prompting"](https://arxiv.org/abs/2205.10625) helps LLMs handle
    larger tasks despite the longer strings (i.e. number of tokens) such inputs and outputs entail;
 * for very large tasks, key pieces of information should be near the beginning or the latter half of
    the text or the LLM may not see the ["needle in the haystack"](https://x.com/GregKamradt/status/1722386725635580292/photo/1);
 * for game-playing and extremely intense reasoning tasks, LLMs can benefit from 
    [seeing small snippets of much larger solutions](https://arxiv.org/abs/2410.09918v1 ) in order to 
    "learn shortcuts" like humans do.

But can LLMs reason _reliably_?  As described in the previous section, most LLMs "sample" or generate
a single _highly-probable_ answer for any given input.  But that generated
answer can be wrong!  In the context of "chain-of-thought" prompting, some
research has explored measures of [self-consistency](https://openreview.net/pdf?id=1PL1NIMMrw),
where the LLM generates **several** explanations (i.e. chain-of-thought answer sequences)
and some simple aggregation (e.g. majority vote on a final component of the answer) can help
boost correctness on benchmarks.  However, the issue of LLM reliability (and thus
LLM _Agent_ reliability) is still an active area of research.

Finally, note that in this section we do **not** seek to define "reasoning" absolutely,
as [researchers are still actively debating whether LLMs actually do reasoning](https://x.com/ylecun/status/1713228046033936717):

<center>
<img 
  alt="Yann on reasoning" 
  src="{{site.baseurl}}/assets/images/yann-reasoning2.jpg"
  width="550px" style="border-radius: 8px; border: 8px solid #ddd;" />
</center>

Instead, here we seek to qualitatively describe useful LLM
phenomena that mimic human reasoning and that (importantly) has given rise to
the impactful LLM Agent work described below.


## LLM Agents

### What is an LLM Agent?

Software agents, such as the 1966 text chatbot [ELIZA](https://en.wikipedia.org/wiki/ELIZA),
interact with human users and/or the wider environment through a [text-based (or similar)
interface](#sw-agents).  These environmental interfaces can include more formal 
[REST APIs](https://en.wikipedia.org/wiki/REST) or other programmatic APIs.  The
interface (or _tool_) must simply accept text as input and return a textual 
representation of the result on the environment.  Software agents encapsulate 
the reasoning and algorithms needed to use these (text-based) interfaces to 
accomplish some goal (e.g. answer a question for a human user).

A rough definition of an **LLM Agent** is a software agent ["that has non-zero 
LLM calls"](#llm-agent-def).  Thus an LLM Agent is simply a software agent that 
leverages modern LLM technology to choose how to interact with the environment.
LLM Agents entail combining the chain-of-thought prompting and reasoning techniques 
described in the previous section with interfaces to the environment--- this 
combination gives the LLM the ability to _act_.  Research systems such as [ReAct](https://arxiv.org/abs/2210.03629)
have demonstrated how this synthesis of LLMs, prompting techniques, and tools gives
rise to LLM Agents that are successful at solving complex tasks:

<center>
<img 
  alt="ReAct from Yao et all" 
  src="{{site.baseurl}}/assets/images/react_overview_yao_et_al2.jpg"
  width="550px" style="border-radius: 8px; border: 8px solid #ddd;" />
</center>

A related thread of current research focuses solely on improving the abilities of LLM Agents to use existing _software-based_
tools and APIs.  [ToolBench](https://github.com/OpenBMB/ToolBench) is one such
influential project seeking to improve how LLMs can achieve fast and effective 
[tool learning](https://arxiv.org/abs/2304.08354).  The [Toolformer](https://arxiv.org/abs/2302.04761) study
furthermore shows how an LLM Agent can reason to choose the best tool among a suite of tools.  Today, many LLM 
services such as OpenAI GPT offer formal [tool APIs](https://platform.openai.com/docs/assistants/tools)
where the user can simply include a text-based description of their own program's
function(s) alongside the input prompt, and the OpenAI LLM may reply with the choice to invoke
those function(s).

 
### Building High-Quality Agents

#### Memory and Retrieval Augmented Generation (RAG)

LLM Agents are most powerful when they have access to a high-fidelity representation
of the environment's current and past states (or *memory*), which can include the results of the LLM Agent's
own actions. For example, an LLM Agent may need access to its lengthy
chat history with an end user.  Furthermore, LLM Agents often benefit from access to ever-expanding 
content from the internet; indeed, some estimates show each day about 15%
of all web searches target [new content](https://danielsgriffin.com/pposts/2023/11/09/true-false-or-useful/#:~:text=%E2%80%9CJust%20some%20history%20on%20this,to%20Google%20even%20through%20today).

Retrieval Augmented Generation (RAG) is one of the most popular and effective ways to 
empower LLM Agents with memory.  While the [seminal work on RAG](https://arxiv.org/pdf/2005.11401) 
outlines a complex, end-to-end-optimized pipeline, many systems today employ
relatively simple [vector databases to serve as a memory component](https://arxiv.org/abs/2312.10997)
of the overall system:

<center>
<img 
  alt="Retrieval-Augmented-Generation with Vector Database from Gao et all" 
  src="{{site.baseurl}}/assets/images/rag-with-index2.jpg"
  width="550px" style="border-radius: 8px; border: 8px solid #ddd;" />
</center>

A RAG system will first ingest "documents" (i.e. chat history, content from the internet, or
other useful data) and encode them into [embedding vectors](https://en.wikipedia.org/wiki/Word_embedding) 
using a model very similar to an LLM (i.e. an LLM with a somewhat different "inference" procedure). 
At query time, the LLM Agent will compute similar embedding vector(s) for the query and match these
vector(s) against all vectors in the database.  (Note that this [vector search](https://en.wikipedia.org/wiki/Vector_database)
operation is well-studied, is very fast, and [there are ubiquitous implementations 
available](#vector-db)).  The matched vectors thus indicate relevant "documents" 
(e.g. specific chat history messages, individual web page snippets, etc.) that should be included 
as context to the LLM that generates the agent's final response (and/or action).  Therefore, 
the "retrieved" documents are added to the LLM input to "augment" the "generated" final response.



#### Optimizing Agents

For most supervised learning systems (including LLMs), the most effective way to 
optimize performance is to grow the training data set and to re-train the model. 
However, this approach is cost prohibitive for most LLM Agent developers
because:
 * basic training [can cost as little as tens of US Dollars (USD) when done very, very efficiently](https://www.youtube.com/watch?v=l8pRSuU81PU) 
    but more typical, extensive training typically costs at least thousands to millions of USD (for GPU hardware, compute, and electricity);
 * training data can be expensive, and the particular training datasets for top-performing LLMs
    (such as GPT-4) are not publicly available;
 * re-training can sometimes fail (due to e.g. [overfitting](https://en.wikipedia.org/wiki/Overfitting)) 
     and requires a distinct skillset to get right.

For LLMs (and in particular LLM Agents) careful tuning of _prompts_ can significantly
improve performance; moreover, researchers have explored _automating_ the process
of prompt tuning.  In particular, [DSPy](https://arxiv.org/html/2404.14544v1) has [successfully helped 
improve](#dspy-success) a variety of agents through a novel blend of 
[prompt engineering](https://learnprompting.org/) and reinforcement learning.  


## Further Reading

The content of this post focuses on topics in the early lectures of the [Fall 2024 UC Berkeley LLM Agents MOOC](https://llmagents-learning.org/f24).  For
more on LLM Agents, consult the course.  For complete LLM Agent systems and frameworks, consult:
  * appendix A of the [AutoGen paper](https://arxiv.org/abs/2308.08155) for a list of modern LLM Agent frameworks;
  * [OpenHands](https://arxiv.org/pdf/2407.16741) for code-generating LLM Agents;
  * [TapeAgents](https://github.com/servicenow/tapeagents) for a timeline-centered LLM Agent framework;
  * [LlamaIndex](https://www.llamaindex.ai/) for an industrial LLM Agent platform.

## Endnotes

1. <a name="illya-30-papers"></a>This post assumes only very basic machine learning knowledge.  For survey of the most cutting edge machine learning research (which includes LLMs and more), see [this collection of papers from Ilya Sutskever](https://arc.net/folder/D0472A20-9C20-4D3F-B145-D2865C0A9FEE).
1. <a name="45tb"></a>OpenAI's [GPT-3 was trained on over 45 Terabytes of text](https://arxiv.org/pdf/2005.14165).
1. <a name="sw-agents"></a>For a detailed discussion of relevant _agents_ in Artificial Intelligence, see [Lecture 2](https://www.youtube.com/watch?v=RM6ZArd2nVc).
1. <a name="llm-agent-def"></a>Jerry Liu of LlamaIndex during [Lecture 3](https://www.youtube.com/live/OOdtmCMSOo4?si=sBqtLtCxOcvHaMiM&t=3516).
1. <a name="vector-db"></a>See for example [`pgvector`](https://github.com/pgvector/pgvector) or [Rockset](https://rockset.com/).
1. <a name="dspy-success"></a>DSPy's success stories include [beating human expert prompt engineers](https://x.com/learnprompting/status/1809301301760537021) as well as empowering
[a University of Toronto team to win the MEDIQA competition](https://arxiv.org/abs/2404.14544).
