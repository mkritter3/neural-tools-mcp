# Delete Invite

## OpenAPI

````yaml delete /v1/organizations/invites/{invite_id}
paths:
  path: /v1/organizations/invites/{invite_id}
  method: delete
  servers:
    - url: https://api.anthropic.com
  request:
    security: []
    parameters:
      path:
        invite_id:
          schema:
            - type: string
              required: true
              title: Invite Id
              description: ID of the Invite.
      query: {}
      header:
        x-api-key:
          schema:
            - type: string
              required: true
              title: X-Api-Key
              description: >-
                Your unique Admin API key for authentication. 


                This key is required in the header of all Admin API requests, to
                authenticate your account and access Anthropic's services. Get
                your Admin API key through the
                [Console](https://console.anthropic.com/settings/admin-keys).
        anthropic-version:
          schema:
            - type: string
              required: true
              title: Anthropic-Version
              description: >-
                The version of the Anthropic API you want to use.


                Read more about versioning and our version history
                [here](https://docs.anthropic.com/en/api/versioning).
      cookie: {}
    body: {}
    codeSamples:
      - lang: bash
        source: >-
          curl --request DELETE
          "https://api.anthropic.com/v1/organizations/invites/invite_015gWxCN9Hfg2QhZwTK7Mdeu"
          \
            --header "anthropic-version: 2023-06-01" \
            --header "content-type: application/json" \
            --header "x-api-key: $ANTHROPIC_ADMIN_KEY"
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              id:
                allOf:
                  - type: string
                    title: Id
                    description: ID of the Invite.
                    examples:
                      - invite_015gWxCN9Hfg2QhZwTK7Mdeu
              type:
                allOf:
                  - type: string
                    enum:
                      - invite_deleted
                    const: invite_deleted
                    title: Type
                    description: |-
                      Deleted object type.

                      For Invites, this is always `"invite_deleted"`.
                    default: invite_deleted
            title: DeleteInviteResponse
            requiredProperties:
              - id
              - type
        examples:
          example:
            value:
              id: invite_015gWxCN9Hfg2QhZwTK7Mdeu
              type: invite_deleted
        description: Successful Response
    4XX:
      application/json:
        schemaArray:
          - type: object
            properties:
              type:
                allOf:
                  - type: string
                    enum:
                      - error
                    const: error
                    title: Type
                    default: error
              error:
                allOf:
                  - oneOf:
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
                    discriminator:
                      propertyName: type
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
            title: ErrorResponse
            requiredProperties:
              - type
              - error
        examples:
          example:
            value:
              type: error
              error:
                type: invalid_request_error
                message: Invalid request
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
        type:
          type: string
          enum:
            - api_error
          const: api_error
          title: Type
          default: api_error
        message:
          type: string
          title: Message
          default: Internal server error
      type: object
      required:
        - type
        - message
      title: APIError
    AuthenticationError:
      properties:
        type:
          type: string
          enum:
            - authentication_error
          const: authentication_error
          title: Type
          default: authentication_error
        message:
          type: string
          title: Message
          default: Authentication error
      type: object
      required:
        - type
        - message
      title: AuthenticationError
    BillingError:
      properties:
        type:
          type: string
          enum:
            - billing_error
          const: billing_error
          title: Type
          default: billing_error
        message:
          type: string
          title: Message
          default: Billing error
      type: object
      required:
        - type
        - message
      title: BillingError
    GatewayTimeoutError:
      properties:
        type:
          type: string
          enum:
            - timeout_error
          const: timeout_error
          title: Type
          default: timeout_error
        message:
          type: string
          title: Message
          default: Request timeout
      type: object
      required:
        - type
        - message
      title: GatewayTimeoutError
    InvalidRequestError:
      properties:
        type:
          type: string
          enum:
            - invalid_request_error
          const: invalid_request_error
          title: Type
          default: invalid_request_error
        message:
          type: string
          title: Message
          default: Invalid request
      type: object
      required:
        - type
        - message
      title: InvalidRequestError
    NotFoundError:
      properties:
        type:
          type: string
          enum:
            - not_found_error
          const: not_found_error
          title: Type
          default: not_found_error
        message:
          type: string
          title: Message
          default: Not found
      type: object
      required:
        - type
        - message
      title: NotFoundError
    OverloadedError:
      properties:
        type:
          type: string
          enum:
            - overloaded_error
          const: overloaded_error
          title: Type
          default: overloaded_error
        message:
          type: string
          title: Message
          default: Overloaded
      type: object
      required:
        - type
        - message
      title: OverloadedError
    PermissionError:
      properties:
        type:
          type: string
          enum:
            - permission_error
          const: permission_error
          title: Type
          default: permission_error
        message:
          type: string
          title: Message
          default: Permission denied
      type: object
      required:
        - type
        - message
      title: PermissionError
    RateLimitError:
      properties:
        type:
          type: string
          enum:
            - rate_limit_error
          const: rate_limit_error
          title: Type
          default: rate_limit_error
        message:
          type: string
          title: Message
          default: Rate limited
      type: object
      required:
        - type
        - message
      title: RateLimitError

````