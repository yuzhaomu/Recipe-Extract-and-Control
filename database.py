from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Recipe(Base):
    __tablename__ = "recipes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), index=True)
    content = Column(Text)
    created_at = Column(String(50))

    ingredients = relationship("Ingredient", back_populates="recipe")
    seasonings = relationship("Seasoning", back_populates="recipe")
    actions = relationship("Action", back_populates="recipe")


class Ingredient(Base):
    __tablename__ = "ingredients"

    id = Column(Integer, primary_key=True, index=True)
    recipe_id = Column(Integer, ForeignKey("recipes.id"))
    name = Column(String(100), index=True)
    quantity = Column(String(50))
    unit = Column(String(20))
    processing_method = Column(String(100))

    recipe = relationship("Recipe", back_populates="ingredients")


class Seasoning(Base):
    __tablename__ = "seasonings"

    id = Column(Integer, primary_key=True, index=True)
    recipe_id = Column(Integer, ForeignKey("recipes.id"))
    name = Column(String(100), index=True)
    quantity = Column(String(50))
    unit = Column(String(20))

    recipe = relationship("Recipe", back_populates="seasonings")


class Action(Base):
    __tablename__ = "actions"

    id = Column(Integer, primary_key=True, index=True)
    recipe_id = Column(Integer, ForeignKey("recipes.id"))
    action_type = Column(String(50), index=True)
    target_type = Column(String(50))
    target_name = Column(String(100))
    duration = Column(String(50))

    recipe = relationship("Recipe", back_populates="actions")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()