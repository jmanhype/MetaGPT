import json
from pydantic import BaseModel, Undefined

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj.__class__.__name__ == 'PydanticUndefinedType':
            return None
        elif isinstance(obj, BaseModel):
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

async def compile_to(self, i, indent=4, ensure_ascii=False):
    try:
        return json.dumps(i, indent=indent, ensure_ascii=ensure_ascii, cls=CustomJSONEncoder)
    except TypeError as e:
        logger.error(f"Serialization error: {e}")
        return json.dumps(str(i), indent=indent, ensure_ascii=ensure_ascii)
