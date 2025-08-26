# Retrieve Message Batch Results

> Streams the results of a Message Batch as a `.jsonl` file.

Each line in the file is a JSON object containing the result of a single request in the Message Batch. Results are not guaranteed to be in the same order as requests. Use the `custom_id` field to match results to requests.

Learn more about the Message Batches API in our [user guide](/en/docs/build-with-claude/batch-processing)

## OpenAPI

````yaml get /v1/messages/batches/{message_batch_id}/results
paths:
  path: /v1/messages/batches/{message_batch_id}/results
  method: get
  servers:
    - url: https://api.anthropic.com
  request:
    security: []
    parameters:
      path:
        message_batch_id:
          schema:
            - type: string
              required: true
              title: Message Batch Id
              description: ID of the Message Batch.
      query: {}
      header:
        anthropic-beta:
          schema:
            - type: array
              items:
                allOf:
                  - type: string
              required: false
              title: Anthropic-Beta
              description: >-
                Optional header to specify the beta version(s) you want to use.


                To use multiple betas, use a comma separated list like
                `beta1,beta2` or specify the header multiple times for each
                beta.
        anthropic-version:
          schema:
            - type: string
              required: true
              title: Anthropic-Version
              description: >-
                The version of the Anthropic API you want to use.


                Read more about versioning and our version history
                [here](https://docs.anthropic.com/en/api/versioning).
        x-api-key:
          schema:
            - type: string
              required: true
              title: X-Api-Key
              description: >-
                Your unique API key for authentication.


                This key is required in the header of all API requests, to
                authenticate your account and access Anthropic's services. Get
                your API key through the
                [Console](https://console.anthropic.com/settings/keys). Each key
                is scoped to a Workspace.
      cookie: {}
    body: {}
    codeSamples:
      - lang: bash
        source: >-
          curl
          https://api.anthropic.com/v1/messages/batches/msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d/results
          \
               --header "x-api-key: $ANTHROPIC_API_KEY" \
               --header "anthropic-version: 2023-06-01"
      - lang: python
        source: |-
          import anthropic

          client = anthropic.Anthropic()

          for result in client.messages.batches.results(
              "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
          ):
              print(result)
      - lang: javascript
        source: |-
          import Anthropic from '@anthropic-ai/sdk';

          const anthropic = new Anthropic();

          for await (const result of await anthropic.messages.batches.results(
            "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
          )) {
            console.log(result);
          }
  response:
    '200':
      application/x-jsonl:
        schemaArray:
          - type: object
            properties:
              custom_id:
                allOf:
                  - description: >-
                      Developer-provided ID created for each request in a
                      Message Batch. Useful for matching results to requests, as
                      results may be given out of request order.


                      Must be unique for each request within the Message Batch.
                    examples:
                      - my-custom-id-1
                    title: Custom Id
                    type: string
              result:
                allOf:
                  - description: >-
                      Processing result for this request.


                      Contains a Message output if processing was successful, an
                      error response if processing failed, or the reason why
                      processing was not attempted, such as cancellation or
                      expiration.
                    discriminator:
                      mapping:
                        canceled: '#/components/schemas/CanceledResult'
                        errored: '#/components/schemas/ErroredResult'
                        expired: '#/components/schemas/ExpiredResult'
                        succeeded: '#/components/schemas/SucceededResult'
                      propertyName: type
                    oneOf:
                      - $ref: '#/components/schemas/SucceededResult'
                      - $ref: '#/components/schemas/ErroredResult'
                      - $ref: '#/components/schemas/CanceledResult'
                      - $ref: '#/components/schemas/ExpiredResult'
            title: MessageBatchIndividualResponse
            description: >-
              This is a single line in the response `.jsonl` file and does not
              represent the response as a whole.
            requiredProperties:
              - custom_id
              - result
        examples:
          example:
            value:
              custom_id: my-custom-id-1
              result:
                message:
                  content:
                    - text: Hi! My name is Claude.
                      type: text
                  id: msg_013Zva2CMHLNnXjNJJKqJ2EF
                  model: claude-sonnet-4-20250514
                  role: assistant
                  stop_reason: end_turn
                  stop_sequence: null
                  type: message
                  usage:
                    input_tokens: 2095
                    output_tokens: 503
                type: succeeded
        description: Successful Response
    4XX:
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - &ref_0
                    discriminator:
                      mapping:
                        api_error: '#/components/schemas/APIError'
                        authentication_error: '#/components/schemas/AuthenticationError'
                        billing_error: '#/components/schemas/BillingError'
                        invalid_request_error: '#/components/schemas/InvalidRequestError'
                        not_found_error: '#/components/schemas/NotFoundError'
                        overloaded_error: '#/components/schemas/OverloadedError'
                        permission_error: '#/components/schemas/PermissionError'
                        rate_limit_error: '#/components/schemas/RateLimitError'
                        timeout_error: '#/components/schemas/GatewayTimeoutError'
                      propertyName: type
                    oneOf:
                      - $ref: '#/components/schemas/InvalidRequestError'
                      - $ref: '#/components/schemas/AuthenticationError'
                      - $ref: '#/components/schemas/BillingError'
                      - $ref: '#/components/schemas/PermissionError'
                      - $ref: '#/components/schemas/NotFoundError'
                      - $ref: '#/components/schemas/RateLimitError'
                      - $ref: '#/components/schemas/GatewayTimeoutError'
                      - $ref: '#/components/schemas/APIError'
                      - $ref: '#/components/schemas/OverloadedError'
                    title: Error
              type:
                allOf:
                  - &ref_1
                    const: error
                    default: error
                    enum:
                      - error
                    title: Type
                    type: string
            title: ErrorResponse
            requiredProperties: &ref_2
              - error
              - type
        examples:
          example:
            value:
              error:
                message: Invalid request
                type: invalid_request_error
              type: error
        description: >-
          Error response.


          See our [errors
          documentation](https://docs.anthropic.com/en/api/errors) for more
          details.
  deprecated: false
  type: path
components:
  schemas:
    APIError:
      properties:
        message:
          default: Internal server error
          title: Message
          type: string
        type:
          const: api_error
          default: api_error
          enum:
            - api_error
          title: Type
          type: string
      required:
        - message
        - type
      title: APIError
      type: object
    AuthenticationError:
      properties:
        message:
          default: Authentication error
          title: Message
          type: string
        type:
          const: authentication_error
          default: authentication_error
          enum:
            - authentication_error
          title: Type
          type: string
      required:
        - message
        - type
      title: AuthenticationError
      type: object
    BillingError:
      properties:
        message:
          default: Billing error
          title: Message
          type: string
        type:
          const: billing_error
          default: billing_error
          enum:
            - billing_error
          title: Type
          type: string
      required:
        - message
        - type
      title: BillingError
      type: object
    CacheCreation:
      properties:
        ephemeral_1h_input_tokens:
          default: 0
          description: The number of input tokens used to create the 1 hour cache entry.
          minimum: 0
          title: Ephemeral 1H Input Tokens
          type: integer
        ephemeral_5m_input_tokens:
          default: 0
          description: The number of input tokens used to create the 5 minute cache entry.
          minimum: 0
          title: Ephemeral 5M Input Tokens
          type: integer
      required:
        - ephemeral_1h_input_tokens
        - ephemeral_5m_input_tokens
      title: CacheCreation
      type: object
    CanceledResult:
      properties:
        type:
          const: canceled
          default: canceled
          enum:
            - canceled
          title: Type
          type: string
      required:
        - type
      title: CanceledResult
      type: object
    CodeExecutionToolResultErrorCode:
      enum:
        - invalid_tool_input
        - unavailable
        - too_many_requests
        - execution_time_exceeded
      title: CodeExecutionToolResultErrorCode
      type: string
    Container:
      description: >-
        Information about the container used in the request (for the code
        execution tool)
      properties:
        expires_at:
          description: The time at which the container will expire.
          format: date-time
          title: Expires At
          type: string
        id:
          description: Identifier for the container used in this request
          title: Id
          type: string
      required:
        - expires_at
        - id
      title: Container
      type: object
    ErrorResponse:
      properties:
        error: *ref_0
        type: *ref_1
      required: *ref_2
      title: ErrorResponse
      type: object
    ErroredResult:
      properties:
        error:
          $ref: '#/components/schemas/ErrorResponse'
        type:
          const: errored
          default: errored
          enum:
            - errored
          title: Type
          type: string
      required:
        - error
        - type
      title: ErroredResult
      type: object
    ExpiredResult:
      properties:
        type:
          const: expired
          default: expired
          enum:
            - expired
          title: Type
          type: string
      required:
        - type
      title: ExpiredResult
      type: object
    GatewayTimeoutError:
      properties:
        message:
          default: Request timeout
          title: Message
          type: string
        type:
          const: timeout_error
          default: timeout_error
          enum:
            - timeout_error
          title: Type
          type: string
      required:
        - message
        - type
      title: GatewayTimeoutError
      type: object
    InvalidRequestError:
      properties:
        message:
          default: Invalid request
          title: Message
          type: string
        type:
          const: invalid_request_error
          default: invalid_request_error
          enum:
            - invalid_request_error
          title: Type
          type: string
      required:
        - message
        - type
      title: InvalidRequestError
      type: object
    Message:
      examples:
        - content:
            - text: Hi! My name is Claude.
              type: text
          id: msg_013Zva2CMHLNnXjNJJKqJ2EF
          model: claude-sonnet-4-20250514
          role: assistant
          stop_reason: end_turn
          stop_sequence: null
          type: message
          usage:
            input_tokens: 2095
            output_tokens: 503
      properties:
        id:
          description: |-
            Unique object identifier.

            The format and length of IDs may change over time.
          examples:
            - msg_013Zva2CMHLNnXjNJJKqJ2EF
          title: Id
          type: string
        type:
          const: message
          default: message
          description: |-
            Object type.

            For Messages, this is always `"message"`.
          enum:
            - message
          title: Type
          type: string
        role:
          const: assistant
          default: assistant
          description: |-
            Conversational role of the generated message.

            This will always be `"assistant"`.
          enum:
            - assistant
          title: Role
          type: string
        content:
          description: >-
            Content generated by the model.


            This is an array of content blocks, each of which has a `type` that
            determines its shape.


            Example:


            ```json

            [{"type": "text", "text": "Hi, I'm Claude."}]

            ```


            If the request input `messages` ended with an `assistant` turn, then
            the response `content` will continue directly from that last turn.
            You can use this to constrain the model's output.


            For example, if the input `messages` were:

            ```json

            [
              {"role": "user", "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"},
              {"role": "assistant", "content": "The best answer is ("}
            ]

            ```


            Then the response `content` might be:


            ```json

            [{"type": "text", "text": "B)"}]

            ```
          examples:
            - - text: Hi! My name is Claude.
                type: text
          items:
            discriminator:
              mapping:
                code_execution_tool_result: '#/components/schemas/ResponseCodeExecutionToolResultBlock'
                container_upload: '#/components/schemas/ResponseContainerUploadBlock'
                mcp_tool_result: '#/components/schemas/ResponseMCPToolResultBlock'
                mcp_tool_use: '#/components/schemas/ResponseMCPToolUseBlock'
                redacted_thinking: '#/components/schemas/ResponseRedactedThinkingBlock'
                server_tool_use: '#/components/schemas/ResponseServerToolUseBlock'
                text: '#/components/schemas/ResponseTextBlock'
                thinking: '#/components/schemas/ResponseThinkingBlock'
                tool_use: '#/components/schemas/ResponseToolUseBlock'
                web_search_tool_result: '#/components/schemas/ResponseWebSearchToolResultBlock'
              propertyName: type
            oneOf:
              - $ref: '#/components/schemas/ResponseTextBlock'
              - $ref: '#/components/schemas/ResponseThinkingBlock'
              - $ref: '#/components/schemas/ResponseRedactedThinkingBlock'
              - $ref: '#/components/schemas/ResponseToolUseBlock'
              - $ref: '#/components/schemas/ResponseServerToolUseBlock'
              - $ref: '#/components/schemas/ResponseWebSearchToolResultBlock'
              - $ref: '#/components/schemas/ResponseCodeExecutionToolResultBlock'
              - $ref: '#/components/schemas/ResponseMCPToolUseBlock'
              - $ref: '#/components/schemas/ResponseMCPToolResultBlock'
              - $ref: '#/components/schemas/ResponseContainerUploadBlock'
          title: Content
          type: array
        model:
          description: The model that handled the request.
          examples:
            - claude-sonnet-4-20250514
          maxLength: 256
          minLength: 1
          title: Model
          type: string
        stop_reason:
          anyOf:
            - enum:
                - end_turn
                - max_tokens
                - stop_sequence
                - tool_use
                - pause_turn
                - refusal
              type: string
            - type: 'null'
          description: >-
            The reason that we stopped.


            This may be one the following values:

            * `"end_turn"`: the model reached a natural stopping point

            * `"max_tokens"`: we exceeded the requested `max_tokens` or the
            model's maximum

            * `"stop_sequence"`: one of your provided custom `stop_sequences`
            was generated

            * `"tool_use"`: the model invoked one or more tools

            * `"pause_turn"`: we paused a long-running turn. You may provide the
            response back as-is in a subsequent request to let the model
            continue.

            * `"refusal"`: when streaming classifiers intervene to handle
            potential policy violations


            In non-streaming mode this value is always non-null. In streaming
            mode, it is null in the `message_start` event and non-null
            otherwise.
          title: Stop Reason
        stop_sequence:
          anyOf:
            - type: string
            - type: 'null'
          default: null
          description: >-
            Which custom stop sequence was generated, if any.


            This value will be a non-null string if one of your custom stop
            sequences was generated.
          title: Stop Sequence
        usage:
          $ref: '#/components/schemas/Usage'
          description: >-
            Billing and rate-limit usage.


            Anthropic's API bills and rate-limits by token counts, as tokens
            represent the underlying cost to our systems.


            Under the hood, the API transforms requests into a format suitable
            for the model. The model's output then goes through a parsing stage
            before becoming an API response. As a result, the token counts in
            `usage` will not match one-to-one with the exact visible content of
            an API request or response.


            For example, `output_tokens` will be non-zero, even for an empty
            string response from Claude.


            Total input tokens in a request is the summation of `input_tokens`,
            `cache_creation_input_tokens`, and `cache_read_input_tokens`.
          examples:
            - input_tokens: 2095
              output_tokens: 503
        container:
          anyOf:
            - $ref: '#/components/schemas/Container'
            - type: 'null'
          default: null
          description: >-
            Information about the container used in this request.


            This will be non-null if a container tool (e.g. code execution) was
            used.
      required:
        - id
        - type
        - role
        - content
        - model
        - stop_reason
        - stop_sequence
        - usage
        - container
      title: Message
      type: object
    NotFoundError:
      properties:
        message:
          default: Not found
          title: Message
          type: string
        type:
          const: not_found_error
          default: not_found_error
          enum:
            - not_found_error
          title: Type
          type: string
      required:
        - message
        - type
      title: NotFoundError
      type: object
    OverloadedError:
      properties:
        message:
          default: Overloaded
          title: Message
          type: string
        type:
          const: overloaded_error
          default: overloaded_error
          enum:
            - overloaded_error
          title: Type
          type: string
      required:
        - message
        - type
      title: OverloadedError
      type: object
    PermissionError:
      properties:
        message:
          default: Permission denied
          title: Message
          type: string
        type:
          const: permission_error
          default: permission_error
          enum:
            - permission_error
          title: Type
          type: string
      required:
        - message
        - type
      title: PermissionError
      type: object
    RateLimitError:
      properties:
        message:
          default: Rate limited
          title: Message
          type: string
        type:
          const: rate_limit_error
          default: rate_limit_error
          enum:
            - rate_limit_error
          title: Type
          type: string
      required:
        - message
        - type
      title: RateLimitError
      type: object
    ResponseCharLocationCitation:
      properties:
        cited_text:
          title: Cited Text
          type: string
        document_index:
          minimum: 0
          title: Document Index
          type: integer
        document_title:
          anyOf:
            - type: string
            - type: 'null'
          title: Document Title
        end_char_index:
          title: End Char Index
          type: integer
        file_id:
          anyOf:
            - type: string
            - type: 'null'
          default: null
          title: File Id
        start_char_index:
          minimum: 0
          title: Start Char Index
          type: integer
        type:
          const: char_location
          default: char_location
          enum:
            - char_location
          title: Type
          type: string
      required:
        - cited_text
        - document_index
        - document_title
        - end_char_index
        - file_id
        - start_char_index
        - type
      title: Character location
      type: object
    ResponseCodeExecutionOutputBlock:
      properties:
        file_id:
          title: File Id
          type: string
        type:
          const: code_execution_output
          default: code_execution_output
          enum:
            - code_execution_output
          title: Type
          type: string
      required:
        - file_id
        - type
      title: ResponseCodeExecutionOutputBlock
      type: object
    ResponseCodeExecutionResultBlock:
      properties:
        content:
          items:
            $ref: '#/components/schemas/ResponseCodeExecutionOutputBlock'
          title: Content
          type: array
        return_code:
          title: Return Code
          type: integer
        stderr:
          title: Stderr
          type: string
        stdout:
          title: Stdout
          type: string
        type:
          const: code_execution_result
          default: code_execution_result
          enum:
            - code_execution_result
          title: Type
          type: string
      required:
        - content
        - return_code
        - stderr
        - stdout
        - type
      title: Code execution result
      type: object
    ResponseCodeExecutionToolResultBlock:
      properties:
        content:
          anyOf:
            - $ref: '#/components/schemas/ResponseCodeExecutionToolResultError'
            - $ref: '#/components/schemas/ResponseCodeExecutionResultBlock'
          title: Content
        tool_use_id:
          pattern: ^srvtoolu_[a-zA-Z0-9_]+$
          title: Tool Use Id
          type: string
        type:
          const: code_execution_tool_result
          default: code_execution_tool_result
          enum:
            - code_execution_tool_result
          title: Type
          type: string
      required:
        - content
        - tool_use_id
        - type
      title: Code execution tool result
      type: object
    ResponseCodeExecutionToolResultError:
      properties:
        error_code:
          $ref: '#/components/schemas/CodeExecutionToolResultErrorCode'
        type:
          const: code_execution_tool_result_error
          default: code_execution_tool_result_error
          enum:
            - code_execution_tool_result_error
          title: Type
          type: string
      required:
        - error_code
        - type
      title: Code execution tool error
      type: object
    ResponseContainerUploadBlock:
      description: Response model for a file uploaded to the container.
      properties:
        file_id:
          title: File Id
          type: string
        type:
          const: container_upload
          default: container_upload
          enum:
            - container_upload
          title: Type
          type: string
      required:
        - file_id
        - type
      title: Container upload
      type: object
    ResponseContentBlockLocationCitation:
      properties:
        cited_text:
          title: Cited Text
          type: string
        document_index:
          minimum: 0
          title: Document Index
          type: integer
        document_title:
          anyOf:
            - type: string
            - type: 'null'
          title: Document Title
        end_block_index:
          title: End Block Index
          type: integer
        file_id:
          anyOf:
            - type: string
            - type: 'null'
          default: null
          title: File Id
        start_block_index:
          minimum: 0
          title: Start Block Index
          type: integer
        type:
          const: content_block_location
          default: content_block_location
          enum:
            - content_block_location
          title: Type
          type: string
      required:
        - cited_text
        - document_index
        - document_title
        - end_block_index
        - file_id
        - start_block_index
        - type
      title: Content block location
      type: object
    ResponseMCPToolResultBlock:
      properties:
        content:
          anyOf:
            - type: string
            - items:
                $ref: '#/components/schemas/ResponseTextBlock'
              type: array
          title: Content
        is_error:
          default: false
          title: Is Error
          type: boolean
        tool_use_id:
          pattern: ^[a-zA-Z0-9_-]+$
          title: Tool Use Id
          type: string
        type:
          const: mcp_tool_result
          default: mcp_tool_result
          enum:
            - mcp_tool_result
          title: Type
          type: string
      required:
        - content
        - is_error
        - tool_use_id
        - type
      title: MCP tool result
      type: object
    ResponseMCPToolUseBlock:
      properties:
        id:
          pattern: ^[a-zA-Z0-9_-]+$
          title: Id
          type: string
        input:
          title: Input
          type: object
        name:
          description: The name of the MCP tool
          title: Name
          type: string
        server_name:
          description: The name of the MCP server
          title: Server Name
          type: string
        type:
          const: mcp_tool_use
          default: mcp_tool_use
          enum:
            - mcp_tool_use
          title: Type
          type: string
      required:
        - id
        - input
        - name
        - server_name
        - type
      title: MCP tool use
      type: object
    ResponsePageLocationCitation:
      properties:
        cited_text:
          title: Cited Text
          type: string
        document_index:
          minimum: 0
          title: Document Index
          type: integer
        document_title:
          anyOf:
            - type: string
            - type: 'null'
          title: Document Title
        end_page_number:
          title: End Page Number
          type: integer
        file_id:
          anyOf:
            - type: string
            - type: 'null'
          default: null
          title: File Id
        start_page_number:
          minimum: 1
          title: Start Page Number
          type: integer
        type:
          const: page_location
          default: page_location
          enum:
            - page_location
          title: Type
          type: string
      required:
        - cited_text
        - document_index
        - document_title
        - end_page_number
        - file_id
        - start_page_number
        - type
      title: Page location
      type: object
    ResponseRedactedThinkingBlock:
      properties:
        data:
          title: Data
          type: string
        type:
          const: redacted_thinking
          default: redacted_thinking
          enum:
            - redacted_thinking
          title: Type
          type: string
      required:
        - data
        - type
      title: Redacted thinking
      type: object
    ResponseSearchResultLocationCitation:
      properties:
        cited_text:
          title: Cited Text
          type: string
        end_block_index:
          title: End Block Index
          type: integer
        search_result_index:
          minimum: 0
          title: Search Result Index
          type: integer
        source:
          title: Source
          type: string
        start_block_index:
          minimum: 0
          title: Start Block Index
          type: integer
        title:
          anyOf:
            - type: string
            - type: 'null'
          title: Title
        type:
          const: search_result_location
          default: search_result_location
          enum:
            - search_result_location
          title: Type
          type: string
      required:
        - cited_text
        - end_block_index
        - search_result_index
        - source
        - start_block_index
        - title
        - type
      title: ResponseSearchResultLocationCitation
      type: object
    ResponseServerToolUseBlock:
      properties:
        id:
          pattern: ^srvtoolu_[a-zA-Z0-9_]+$
          title: Id
          type: string
        input:
          title: Input
          type: object
        name:
          enum:
            - web_search
            - code_execution
          title: Name
          type: string
        type:
          const: server_tool_use
          default: server_tool_use
          enum:
            - server_tool_use
          title: Type
          type: string
      required:
        - id
        - input
        - name
        - type
      title: Server tool use
      type: object
    ResponseTextBlock:
      properties:
        citations:
          anyOf:
            - items:
                discriminator:
                  mapping:
                    char_location: '#/components/schemas/ResponseCharLocationCitation'
                    content_block_location: '#/components/schemas/ResponseContentBlockLocationCitation'
                    page_location: '#/components/schemas/ResponsePageLocationCitation'
                    search_result_location: '#/components/schemas/ResponseSearchResultLocationCitation'
                    web_search_result_location: >-
                      #/components/schemas/ResponseWebSearchResultLocationCitation
                  propertyName: type
                oneOf:
                  - $ref: '#/components/schemas/ResponseCharLocationCitation'
                  - $ref: '#/components/schemas/ResponsePageLocationCitation'
                  - $ref: '#/components/schemas/ResponseContentBlockLocationCitation'
                  - $ref: >-
                      #/components/schemas/ResponseWebSearchResultLocationCitation
                  - $ref: '#/components/schemas/ResponseSearchResultLocationCitation'
              type: array
            - type: 'null'
          default: null
          description: >-
            Citations supporting the text block.


            The type of citation returned will depend on the type of document
            being cited. Citing a PDF results in `page_location`, plain text
            results in `char_location`, and content document results in
            `content_block_location`.
          title: Citations
        text:
          maxLength: 5000000
          minLength: 0
          title: Text
          type: string
        type:
          const: text
          default: text
          enum:
            - text
          title: Type
          type: string
      required:
        - citations
        - text
        - type
      title: Text
      type: object
    ResponseThinkingBlock:
      properties:
        signature:
          title: Signature
          type: string
        thinking:
          title: Thinking
          type: string
        type:
          const: thinking
          default: thinking
          enum:
            - thinking
          title: Type
          type: string
      required:
        - signature
        - thinking
        - type
      title: Thinking
      type: object
    ResponseToolUseBlock:
      properties:
        id:
          pattern: ^[a-zA-Z0-9_-]+$
          title: Id
          type: string
        input:
          title: Input
          type: object
        name:
          minLength: 1
          title: Name
          type: string
        type:
          const: tool_use
          default: tool_use
          enum:
            - tool_use
          title: Type
          type: string
      required:
        - id
        - input
        - name
        - type
      title: Tool use
      type: object
    ResponseWebSearchResultBlock:
      properties:
        encrypted_content:
          title: Encrypted Content
          type: string
        page_age:
          anyOf:
            - type: string
            - type: 'null'
          default: null
          title: Page Age
        title:
          title: Title
          type: string
        type:
          const: web_search_result
          default: web_search_result
          enum:
            - web_search_result
          title: Type
          type: string
        url:
          title: Url
          type: string
      required:
        - encrypted_content
        - page_age
        - title
        - type
        - url
      title: ResponseWebSearchResultBlock
      type: object
    ResponseWebSearchResultLocationCitation:
      properties:
        cited_text:
          title: Cited Text
          type: string
        encrypted_index:
          title: Encrypted Index
          type: string
        title:
          anyOf:
            - maxLength: 512
              type: string
            - type: 'null'
          title: Title
        type:
          const: web_search_result_location
          default: web_search_result_location
          enum:
            - web_search_result_location
          title: Type
          type: string
        url:
          title: Url
          type: string
      required:
        - cited_text
        - encrypted_index
        - title
        - type
        - url
      title: ResponseWebSearchResultLocationCitation
      type: object
    ResponseWebSearchToolResultBlock:
      properties:
        content:
          anyOf:
            - $ref: '#/components/schemas/ResponseWebSearchToolResultError'
            - items:
                $ref: '#/components/schemas/ResponseWebSearchResultBlock'
              type: array
          title: Content
        tool_use_id:
          pattern: ^srvtoolu_[a-zA-Z0-9_]+$
          title: Tool Use Id
          type: string
        type:
          const: web_search_tool_result
          default: web_search_tool_result
          enum:
            - web_search_tool_result
          title: Type
          type: string
      required:
        - content
        - tool_use_id
        - type
      title: Web search tool result
      type: object
    ResponseWebSearchToolResultError:
      properties:
        error_code:
          $ref: '#/components/schemas/WebSearchToolResultErrorCode'
        type:
          const: web_search_tool_result_error
          default: web_search_tool_result_error
          enum:
            - web_search_tool_result_error
          title: Type
          type: string
      required:
        - error_code
        - type
      title: ResponseWebSearchToolResultError
      type: object
    ServerToolUsage:
      properties:
        web_search_requests:
          default: 0
          description: The number of web search tool requests.
          examples:
            - 0
          minimum: 0
          title: Web Search Requests
          type: integer
      required:
        - web_search_requests
      title: ServerToolUsage
      type: object
    SucceededResult:
      properties:
        message:
          $ref: '#/components/schemas/Message'
        type:
          const: succeeded
          default: succeeded
          enum:
            - succeeded
          title: Type
          type: string
      required:
        - message
        - type
      title: SucceededResult
      type: object
    Usage:
      properties:
        cache_creation:
          anyOf:
            - $ref: '#/components/schemas/CacheCreation'
            - type: 'null'
          default: null
          description: Breakdown of cached tokens by TTL
        cache_creation_input_tokens:
          anyOf:
            - minimum: 0
              type: integer
            - type: 'null'
          default: null
          description: The number of input tokens used to create the cache entry.
          examples:
            - 2051
          title: Cache Creation Input Tokens
        cache_read_input_tokens:
          anyOf:
            - minimum: 0
              type: integer
            - type: 'null'
          default: null
          description: The number of input tokens read from the cache.
          examples:
            - 2051
          title: Cache Read Input Tokens
        input_tokens:
          description: The number of input tokens which were used.
          examples:
            - 2095
          minimum: 0
          title: Input Tokens
          type: integer
        output_tokens:
          description: The number of output tokens which were used.
          examples:
            - 503
          minimum: 0
          title: Output Tokens
          type: integer
        server_tool_use:
          anyOf:
            - $ref: '#/components/schemas/ServerToolUsage'
            - type: 'null'
          default: null
          description: The number of server tool requests.
        service_tier:
          anyOf:
            - enum:
                - standard
                - priority
                - batch
              type: string
            - type: 'null'
          default: null
          description: If the request used the priority, standard, or batch tier.
          title: Service Tier
      required:
        - cache_creation
        - cache_creation_input_tokens
        - cache_read_input_tokens
        - input_tokens
        - output_tokens
        - server_tool_use
        - service_tier
      title: Usage
      type: object
    WebSearchToolResultErrorCode:
      enum:
        - invalid_tool_input
        - unavailable
        - max_uses_exceeded
        - too_many_requests
        - query_too_long
      title: WebSearchToolResultErrorCode
      type: string

````