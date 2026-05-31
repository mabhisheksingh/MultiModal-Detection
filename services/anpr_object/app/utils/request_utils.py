from typing import Optional, Any, Dict

anpr_supported_input_types = [
    "video_url",
    "image_url",
    "video_file",
    "image_file",
]


class RequestTraceUtils:
    @staticmethod
    def build_triton_request_id(
        request_id: Optional[str],
        source_name: Optional[str],
        frame_idx: int,
    ) -> Optional[str]:
        if not request_id:
            return None
        normalized_source = (
            (source_name or "unknown_source")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )
        return f"{request_id}:{normalized_source}:{frame_idx}"


def model_dump_compat(model_obj: Any) -> Dict[str, Any]:
    if model_obj is None:
        return {}
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump(exclude_none=True)
    return dict(model_obj)
