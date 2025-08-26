# List Invites

## OpenAPI

````yaml get /v1/organizations/invites
paths:
  path: /v1/organizations/invites
  method: get
  servers:
    - url: https://api.anthropic.com
  request:
    security: []
    parameters:
      path: {}
      query:
        before_id:
          schema:
            - type: string
              required: false
              title: Before Id
              description: >-
                ID of the object to use as a cursor for pagination. When
                provided, returns the page of results immediately before this
                object.
        after_id:
          schema:
            - type: string
              required: false
              title: After Id
              description: >-
                ID of the object to use as a cursor for pagination. When
                provided, returns the page of results immediately after this
                object.
        limit:
          schema:
            - type: integer
              required: false
              title: Limit
              description: |-
                Number of items to return per page.

                Defaults to `20`. Ranges from `1` to `1000`.
              maximum: 1000
              minimum: 1
              default: 20
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
        source: |-
          curl https://api.anthropic.com/v1/organizations/invites \
               --header "x-api-key: $ANTHROPIC_ADMIN_KEY" \
               --header "anthropic-version: 2023-06-01"
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              data:
                allOf:
                  - items:
                      $ref: '#/components/schemas/InviteSchema'
                    type: array
                    title: Data
              has_more:
                allOf:
                  - type: boolean
                    title: Has More
                    description: >-
                      Indicates if there are more results in the requested page
                      direction.
              first_id:
                allOf:
                  - anyOf:
                      - type: string
                      - type: 'null'
                    title: First Id
                    description: >-
                      First ID in the `data` list. Can be used as the
                      `before_id` for the previous page.
              last_id:
                allOf:
                  - anyOf:
                      - type: string
                      - type: 'null'
                    title: Last Id
                    description: >-
                      Last ID in the `data` list. Can be used as the `after_id`
                      for the next page.
            title: ListResponse[InviteSchema]
            requiredProperties:
              - data
              - has_more
              - first_id
              - last_id
        examples:
          example:
            value:
              data:
                - id: invite_015gWxCN9Hfg2QhZwTK7Mdeu
                  type: invite
                  email: user@emaildomain.com
                  role: user
                  invited_at: '2024-10-30T23:58:27.427722Z'
                  expires_at: '2024-11-20T23:58:27.427722Z'
                  status: pending
              has_more: true
              first_id: <string>
              last_id: <string>
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
    InviteSchema:
      properties:
        id:
          type: string
          title: Id
          description: ID of the Invite.
          examples:
            - invite_015gWxCN9Hfg2QhZwTK7Mdeu
        type:
          type: string
          enum:
            - invite
          const: invite
          title: Type
          description: |-
            Object type.

            For Invites, this is always `"invite"`.
          default: invite
        email:
          type: string
          title: Email
          description: Email of the User being invited.
          examples:
            - user@emaildomain.com
        role:
          allOf:
            - $ref: '#/components/schemas/RoleSchema'
          description: Organization role of the User.
          examples:
            - user
            - developer
            - billing
            - admin
        invited_at:
          type: string
          format: date-time
          title: Invited At
          description: RFC 3339 datetime string indicating when the Invite was created.
          examples:
            - '2024-10-30T23:58:27.427722Z'
        expires_at:
          type: string
          format: date-time
          title: Expires At
          description: RFC 3339 datetime string indicating when the Invite expires.
          examples:
            - '2024-11-20T23:58:27.427722Z'
        status:
          allOf:
            - $ref: '#/components/schemas/InviteStatusSchema'
          description: Status of the Invite.
          examples:
            - pending
      type: object
      required:
        - id
        - type
        - email
        - role
        - invited_at
        - expires_at
        - status
      title: InviteSchema
    InviteStatusSchema:
      type: string
      enum:
        - accepted
        - expired
        - deleted
        - pending
      title: InviteStatusSchema
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
    RoleSchema:
      type: string
      enum:
        - user
        - developer
        - billing
        - admin
      title: RoleSchema

````