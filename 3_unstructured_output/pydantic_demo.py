from pydantic import BaseModel, EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email :EmailStr
    gpa: Optional[float] = Field(None, ge=0.0, le=10.0)

new_student = {'name':"Bob",
               'age':'22',
               'email':'bob@gmsil.com',
               'gpa':'3.5'}

student = Student(**new_student)

student_dict = student.dict()

print(student_dict['age'])

student_json = student.json()