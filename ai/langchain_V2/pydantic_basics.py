from pydantic import BaseModel


class Chunk(BaseModel):
    text: str
    score: float


chunk = Chunk(text="Python is a programming language", score=0.91)

print(chunk)
print(chunk.text)
print(chunk.score)


from pydantic import BaseModel, Field, ValidationError


# -----------------------------
# 1. Define the data structure
# -----------------------------


class User(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool = True
    skills: list[str]


user = User(
    name="Arun", age=25, email="arun@example.com", skills=["Python", "LangChain", "RAG"]
)


print("Name:", user.name)
print("Age:", user.age)
print("Email:", user.email)
print("Active:", user.is_active)
print("Skills:", user.skills)


user_data = user.model_dump()
print(user_data)


user_json = user.model_dump_json()
print(user_json)


try:
    invalid_user = User(
        name="John", age="hello", email="john@example.com", skills=["Python"]
    )

except ValidationError as error:
    print("\nValidation Error:")
    print(error)
