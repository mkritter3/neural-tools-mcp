# Models overview

> Claude is a family of state-of-the-art large language models developed by Anthropic. This guide introduces our models and compares their performance with legacy models. 

<Tip>The next generation of Claude Models:<br /><br />
**Claude Opus 4.1** - Our most capable and intelligent model yet. Claude Opus 4.1 sets new standards in complex reasoning and advanced coding. [Learn more](https://www.anthropic.com/news/claude-opus-4-1).<br /><br />
**Claude Sonnet 4** - Our high-performance model with exceptional reasoning and efficiency. [Learn more](https://www.anthropic.com/news/claude-4).</Tip>

<CardGroup cols={2}>
  <Card title="Claude Opus 4.1" icon="trophy" href="/en/docs/about-claude/models/overview#model-comparison-table">
    Our most powerful and capable model

    * <Icon icon="inbox-in" iconType="thin" /> Text and image input
    * <Icon icon="inbox-out" iconType="thin" /> Text output
    * <Icon icon="book" iconType="thin" /> 200k context window
    * <Icon icon="brain" iconType="thin" /> Superior reasoning capabilities
  </Card>

  <Card title="Claude Sonnet 4" icon="star" href="/en/docs/about-claude/models/overview#model-comparison-table">
    High-performance model with exceptional reasoning capabilities

    * <Icon icon="inbox-in" iconType="thin" /> Text and image input
    * <Icon icon="inbox-out" iconType="thin" /> Text output
    * <Icon icon="book" iconType="thin" /> 200k context window (1M context beta available)
  </Card>
</CardGroup>

***

## Model names

| Model             | Anthropic API                                             | AWS Bedrock                                 | GCP Vertex AI                |
| ----------------- | --------------------------------------------------------- | ------------------------------------------- | ---------------------------- |
| Claude Opus 4.1   | `claude-opus-4-1-20250805`                                | `anthropic.claude-opus-4-1-20250805-v1:0`   | `claude-opus-4-1@20250805`   |
| Claude Opus 4     | `claude-opus-4-20250514`                                  | `anthropic.claude-opus-4-20250514-v1:0`     | `claude-opus-4@20250514`     |
| Claude Sonnet 4   | `claude-sonnet-4-20250514`                                | `anthropic.claude-sonnet-4-20250514-v1:0`   | `claude-sonnet-4@20250514`   |
| Claude Sonnet 3.7 | `claude-3-7-sonnet-20250219` (`claude-3-7-sonnet-latest`) | `anthropic.claude-3-7-sonnet-20250219-v1:0` | `claude-3-7-sonnet@20250219` |
| Claude Haiku 3.5  | `claude-3-5-haiku-20241022` (`claude-3-5-haiku-latest`)   | `anthropic.claude-3-5-haiku-20241022-v1:0`  | `claude-3-5-haiku@20241022`  |
| Claude Haiku 3    | `claude-3-haiku-20240307`                                 | `anthropic.claude-3-haiku-20240307-v1:0`    | `claude-3-haiku@20240307`    |

<Note>Models with the same snapshot date (e.g., 20240620) are identical across all platforms and do not change. The snapshot date in the model name ensures consistency and allows developers to rely on stable performance across different environments.</Note>

<Note>
  Opus 4.1 does not allow both `temperature` and `top_p` parameters to be specified. Please use only one.
</Note>

### Model aliases

For convenience during development and testing, we offer aliases for our model ids. These aliases automatically point to the most recent snapshot of a given model. When we release new model snapshots, we migrate aliases to point to the newest version of a model, typically within a week of the new release.

<Tip>
  While aliases are useful for experimentation, we recommend using specific model versions (e.g., `claude-sonnet-4-20250514`) in production applications to ensure consistent behavior.
</Tip>

| Model             | Alias                      | Model ID                     |
| ----------------- | -------------------------- | ---------------------------- |
| Claude Opus 4.1   | `claude-opus-4-1`          | `claude-opus-4-1-20250805`   |
| Claude Opus 4     | `claude-opus-4-0`          | `claude-opus-4-20250514`     |
| Claude Sonnet 4   | `claude-sonnet-4-0`        | `claude-sonnet-4-20250514`   |
| Claude Sonnet 3.7 | `claude-3-7-sonnet-latest` | `claude-3-7-sonnet-20250219` |
| Claude Haiku 3.5  | `claude-3-5-haiku-latest`  | `claude-3-5-haiku-20241022`  |

<Note>
  Aliases are subject to the same rate limits and pricing as the underlying model version they reference.
</Note>

### Model comparison table

To help you choose the right model for your needs, we've compiled a table comparing the key features and capabilities of each model in the Claude family:

| Feature                                                               | Claude Opus 4.1                                                                                      | Claude Opus 4                                                                                        | Claude Sonnet 4                                                                                       | Claude Sonnet 3.7                                                                                     | Claude Haiku 3.5                                                                                       | Claude Haiku 3                                                                                       |
| :-------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Description**                                                       | Our most capable model                                                                               | Our previous flagship model                                                                          | High-performance model                                                                                | High-performance model with early extended thinking                                                   | Our fastest model                                                                                      | Fast and compact model for near-instant responsiveness                                               |
| **Strengths**                                                         | Highest level of intelligence and capability                                                         | Very high intelligence and capability                                                                | High intelligence and balanced performance                                                            | High intelligence with toggleable extended thinking                                                   | Intelligence at blazing speeds                                                                         | Quick and accurate targeted performance                                                              |
| **Multilingual**                                                      | Yes                                                                                                  | Yes                                                                                                  | Yes                                                                                                   | Yes                                                                                                   | Yes                                                                                                    | Yes                                                                                                  |
| **Vision**                                                            | Yes                                                                                                  | Yes                                                                                                  | Yes                                                                                                   | Yes                                                                                                   | Yes                                                                                                    | Yes                                                                                                  |
| **[Extended thinking](/en/docs/build-with-claude/extended-thinking)** | Yes                                                                                                  | Yes                                                                                                  | Yes                                                                                                   | Yes                                                                                                   | No                                                                                                     | No                                                                                                   |
| **[Priority Tier](/en/api/service-tiers)**                            | Yes                                                                                                  | Yes                                                                                                  | Yes                                                                                                   | Yes                                                                                                   | Yes                                                                                                    | No                                                                                                   |
| **API model name**                                                    | `claude-opus-4-1-20250805`                                                                           | `claude-opus-4-20250514`                                                                             | `claude-sonnet-4-20250514`                                                                            | `claude-3-7-sonnet-20250219`                                                                          | `claude-3-5-haiku-20241022`                                                                            | `claude-3-haiku-20240307`                                                                            |
| **Comparative latency**                                               | Moderately Fast                                                                                      | Moderately Fast                                                                                      | Fast                                                                                                  | Fast                                                                                                  | Fastest                                                                                                | Fast                                                                                                 |
| **Context window**                                                    | <Tooltip tip="~150K words \ ~680K unicode characters">200K</Tooltip>                                 | <Tooltip tip="~150K words \ ~680K unicode characters">200K</Tooltip>                                 | <Tooltip tip="~150K words \ ~680K unicode characters">200K</Tooltip> / <br /> 1M (beta)<sup>2</sup>   | <Tooltip tip="~150K words \ ~680K unicode characters">200K</Tooltip>                                  | <Tooltip tip="~150K words \ ~215K unicode characters">200K</Tooltip>                                   | <Tooltip tip="~150K words \ ~680K unicode characters">200K</Tooltip>                                 |
| **Max output**                                                        | <Tooltip tip="~24K words \ 109K unicode characters \ ~50 single spaced pages">32000 tokens</Tooltip> | <Tooltip tip="~24K words \ 109K unicode characters \ ~50 single spaced pages">32000 tokens</Tooltip> | <Tooltip tip="~48K words \ 218K unicode characters \ ~100 single spaced pages">64000 tokens</Tooltip> | <Tooltip tip="~48K words \ 218K unicode characters \ ~100 single spaced pages">64000 tokens</Tooltip> | <Tooltip tip="~6.2K words \ 28K unicode characters \ ~12-13 single spaced pages">8192 tokens</Tooltip> | <Tooltip tip="~3.1K words \ 14K unicode characters \ ~6-7 single spaced pages">4096 tokens</Tooltip> |
| **Training data cut-off**                                             | Mar 2025                                                                                             | Mar 2025                                                                                             | Mar 2025                                                                                              | Nov 2024<sup>1</sup>                                                                                  | July 2024                                                                                              | Aug 2023                                                                                             |

*<sup>1 - While trained on publicly available information on the internet through November 2024, Claude Sonnet 3.7's knowledge cut-off date is the end of October 2024. This means the models' knowledge base is most extensive and reliable on information and events up to October 2024.</sup>*

*<sup>2 - Claude Sonnet 4 supports a [1M token context window](/en/docs/build-with-claude/context-windows#1m-token-context-window) when using the `context-1m-2025-08-07` beta header. [Long context pricing](/en/docs/about-claude/pricing#long-context-pricing) applies to requests exceeding 200K tokens.</sup>*

<Note>
  Include the beta header `output-128k-2025-02-19` in your API request to increase the maximum output token length to 128k tokens for Claude Sonnet 3.7.

  We strongly suggest using our [streaming Messages API](/en/docs/build-with-claude/streaming) to avoid timeouts when generating longer outputs.
  See our guidance on [long requests](/en/api/errors#long-requests) for more details.
</Note>

### Model pricing

The table below shows the price per million tokens for each model:

| Model                                                                      | Base Input Tokens | 5m Cache Writes | 1h Cache Writes | Cache Hits & Refreshes | Output Tokens |
| -------------------------------------------------------------------------- | ----------------- | --------------- | --------------- | ---------------------- | ------------- |
| Claude Opus 4.1                                                            | \$15 / MTok       | \$18.75 / MTok  | \$30 / MTok     | \$1.50 / MTok          | \$75 / MTok   |
| Claude Opus 4                                                              | \$15 / MTok       | \$18.75 / MTok  | \$30 / MTok     | \$1.50 / MTok          | \$75 / MTok   |
| Claude Sonnet 4                                                            | \$3 / MTok        | \$3.75 / MTok   | \$6 / MTok      | \$0.30 / MTok          | \$15 / MTok   |
| Claude Sonnet 3.7                                                          | \$3 / MTok        | \$3.75 / MTok   | \$6 / MTok      | \$0.30 / MTok          | \$15 / MTok   |
| Claude Sonnet 3.5 ([deprecated](/en/docs/about-claude/model-deprecations)) | \$3 / MTok        | \$3.75 / MTok   | \$6 / MTok      | \$0.30 / MTok          | \$15 / MTok   |
| Claude Haiku 3.5                                                           | \$0.80 / MTok     | \$1 / MTok      | \$1.6 / MTok    | \$0.08 / MTok          | \$4 / MTok    |
| Claude Opus 3 ([deprecated](/en/docs/about-claude/model-deprecations))     | \$15 / MTok       | \$18.75 / MTok  | \$30 / MTok     | \$1.50 / MTok          | \$75 / MTok   |
| Claude Haiku 3                                                             | \$0.25 / MTok     | \$0.30 / MTok   | \$0.50 / MTok   | \$0.03 / MTok          | \$1.25 / MTok |

## Prompt and output performance

Claude 4 models excel in:

* **Performance**: Top-tier results in reasoning, coding, multilingual tasks, long-context handling, honesty, and image processing. See the [Claude 4 blog post](http://www.anthropic.com/news/claude-4) for more information.
* **Engaging responses**: Claude models are ideal for applications that require rich, human-like interactions.

  * If you prefer more concise responses, you can adjust your prompts to guide the model toward the desired output length. Refer to our [prompt engineering guides](/en/docs/build-with-claude/prompt-engineering) for details.
  * For specific Claude 4 prompting best practices, see our [Claude 4 best practices guide](/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices).
* **Output quality**: When migrating from previous model generations to Claude 4, you may notice larger improvements in overall performance.

## Migrating to Claude 4

In most cases, you can switch from Claude 3.7 models to Claude 4 models with minimal changes:

1. Update your model name:
   * From: `claude-3-7-sonnet-20250219`
   * To: `claude-sonnet-4-20250514` or `claude-opus-4-1-20250805`

2. Your existing API calls will continue to work without modification, although API behavior has changed slightly in Claude 4 models (see [API release notes](/en/release-notes/api) for details).

For more details, see [Migrating to Claude 4](/en/docs/about-claude/models/migrating-to-claude-4).

***

## Get started with Claude

If you're ready to start exploring what Claude can do for you, let's dive in! Whether you're a developer looking to integrate Claude into your applications or a user wanting to experience the power of AI firsthand, we've got you covered.

<Note>Looking to chat with Claude? Visit [claude.ai](http://www.claude.ai)!</Note>

<CardGroup cols={3}>
  <Card title="Intro to Claude" icon="check" href="/en/docs/intro-to-claude">
    Explore Claude’s capabilities and development flow.
  </Card>

  <Card title="Quickstart" icon="bolt-lightning" href="/en/resources/quickstarts">
    Learn how to make your first API call in minutes.
  </Card>

  <Card title="Anthropic Console" icon="code" href="https://console.anthropic.com">
    Craft and test powerful prompts directly in your browser.
  </Card>
</CardGroup>

If you have any questions or need assistance, don't hesitate to reach out to our [support team](https://support.anthropic.com/) or consult the [Discord community](https://www.anthropic.com/discord).
