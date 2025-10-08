from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

password = "mypassword123"
hashed = pwd_context.hash(password)

print("Hashed password:", hashed)
