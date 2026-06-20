class APIConstants:
    REQUEST_ID_HEADER: str = "X-Request-ID"
    REQUEST_ID_HEADER_PARAM: str = "x-request-id"
    DEFAULT_REQUEST_ID: str = "request_without_id"


ERROR_RESPONSES = {
    400: {"description": "Bad request"},
    500: {"description": "Internal server error"},
}
