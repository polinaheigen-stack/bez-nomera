from .service import ServiceError

# Small public plugin API; provider plugins use the same JSON error contract.
ApiError = ServiceError
