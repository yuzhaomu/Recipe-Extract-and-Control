from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
import os

from database import init_db, get_db, Recipe, Ingredient, Seasoning, Action
from inference import RecipeInference

inference = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference
    inference = RecipeInference()
    yield

app = FastAPI(title="智能菜谱系统", description="基于NER和RE模型的菜谱信息抽取系统", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

class RecipeInput(BaseModel):
    text: str
    title: str = None

class RecipeResponse(BaseModel):
    id: int
    title: str
    text: str
    entities: list
    relations: list

class EntityResponse(BaseModel):
    id: int
    text: str
    label: str
    start: int
    end: int

class RelationResponse(BaseModel):
    head: str
    head_label: str
    tail: str
    tail_label: str
    relation: str


@app.get("/")
def read_root():
    return {"message": "智能菜谱系统 API"}


@app.post("/extract", response_model=dict)
def extract_recipe(input_data: RecipeInput):
    if not inference:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    result = inference.extract_recipe(input_data.text)
    
    return {
        "text": result["text"],
        "entities": result["entities"],
        "relations": result["relations"]
    }


@app.post("/recipe", response_model=RecipeResponse)
def create_recipe(input_data: RecipeInput, db: Session = Depends(get_db)):
    if not inference:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    result = inference.extract_recipe(input_data.text)
    
    title = input_data.title or "未命名菜谱"
    
    recipe = Recipe(
        title=title,
        content=input_data.text,
        created_at=datetime.now().isoformat()
    )
    db.add(recipe)
    db.commit()
    db.refresh(recipe)
    
    existing_ingredients = set()
    existing_seasonings = set()
    existing_actions = set()
    
    for entity in result["entities"]:
        if entity["label"] == "食材":
            name = entity["text"]
            if name not in existing_ingredients:
                existing_ingredients.add(name)
                ingredient = Ingredient(
                    recipe_id=recipe.id,
                    name=name,
                    quantity="",
                    unit="",
                    processing_method=""
                )
                db.add(ingredient)
        elif entity["label"] == "调料":
            name = entity["text"]
            if name not in existing_seasonings:
                existing_seasonings.add(name)
                seasoning = Seasoning(
                    recipe_id=recipe.id,
                    name=name,
                    quantity="",
                    unit=""
                )
                db.add(seasoning)
        elif entity["label"] == "动作":
            action_type = entity["text"]
            if action_type not in existing_actions:
                existing_actions.add(action_type)
                action = Action(
                    recipe_id=recipe.id,
                    action_type=action_type,
                    target_type="",
                    target_name="",
                    duration=""
                )
                db.add(action)
    
    for relation in result["relations"]:
        if relation["relation"] == "食材-份量":
            ingredient = db.query(Ingredient).filter(
                Ingredient.recipe_id == recipe.id,
                Ingredient.name == relation["head"]
            ).first()
            if ingredient:
                quantity_text = relation["tail"]
                ingredient.quantity = quantity_text
        elif relation["relation"] == "食材-处理方式":
            ingredient = db.query(Ingredient).filter(
                Ingredient.recipe_id == recipe.id,
                Ingredient.name == relation["head"]
            ).first()
            if ingredient:
                ingredient.processing_method = relation["tail"]
        elif relation["relation"] == "调料-份量":
            seasoning = db.query(Seasoning).filter(
                Seasoning.recipe_id == recipe.id,
                Seasoning.name == relation["head"]
            ).first()
            if seasoning:
                seasoning.quantity = relation["tail"]
        elif relation["relation"] == "动作-对象":
            action = db.query(Action).filter(
                Action.recipe_id == recipe.id,
                Action.action_type == relation["head"]
            ).first()
            if action:
                action.target_type = relation["tail_label"]
                action.target_name = relation["tail"]
        elif relation["relation"] == "动作-时间":
            action = db.query(Action).filter(
                Action.recipe_id == recipe.id,
                Action.action_type == relation["head"]
            ).first()
            if action:
                action.duration = relation["tail"]
    
    db.commit()
    
    return {
        "id": recipe.id,
        "title": recipe.title,
        "text": recipe.content,
        "entities": result["entities"],
        "relations": result["relations"]
    }


@app.get("/recipes")
def get_recipes(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    recipes = db.query(Recipe).offset(skip).limit(limit).all()
    return recipes


@app.get("/recipes/{recipe_id}")
def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="菜谱未找到")
    
    ingredients = db.query(Ingredient).filter(Ingredient.recipe_id == recipe_id).all()
    seasonings = db.query(Seasoning).filter(Seasoning.recipe_id == recipe_id).all()
    actions = db.query(Action).filter(Action.recipe_id == recipe_id).all()
    
    return {
        "id": recipe.id,
        "title": recipe.title,
        "content": recipe.content,
        "created_at": recipe.created_at,
        "ingredients": [
            {
                "name": i.name,
                "quantity": i.quantity,
                "unit": i.unit,
                "processing_method": i.processing_method
            } for i in ingredients
        ],
        "seasonings": [
            {
                "name": s.name,
                "quantity": s.quantity,
                "unit": s.unit
            } for s in seasonings
        ],
        "actions": [
            {
                "action_type": a.action_type,
                "target_type": a.target_type,
                "target_name": a.target_name,
                "duration": a.duration
            } for a in actions
        ]
    }


@app.delete("/recipes/{recipe_id}")
def delete_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="菜谱未找到")
    
    db.delete(recipe)
    db.commit()
    
    return {"message": "菜谱已删除"}


if __name__ == "__main__":
    import uvicorn
    import argparse
    import socket

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    args = parser.parse_args()

    port = args.port
    while port <= 8010:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        result = sock.connect_ex(('0.0.0.0', port))
        sock.close()
        
        if result != 0:
            break
        port += 1

    print(f"服务启动在端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)