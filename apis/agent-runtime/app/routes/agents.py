# app/routes/agents.py
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import RegisteredAgent
from db.session import get_session
from models.agent import AgentTemplateCreate, AgentTemplateResponse
from services.agent_executor import clear_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentTemplateResponse)
def create_agent(template: AgentTemplateCreate, db: Session = Depends(get_session)) -> AgentTemplateResponse:
    existing = db.get(RegisteredAgent, template.agent_id)
    if existing:
        logger.info("Updating existing agent '%s' (template: %s)", template.agent_id, template.template_id)
        existing.template_id = template.template_id
        existing.name = template.name
        existing.description = template.description
        existing.status = "registered"
        existing.loaded = True
        existing.agent_config = template.model_dump()
        existing.metadata_json = template.metadata
        db.add(existing)
        db.commit()
        db.refresh(existing)

        clear_cache(existing.agent_id)

        return AgentTemplateResponse(
            agent_id=existing.agent_id,
            template_id=existing.template_id,
            status=existing.status,
            loaded=existing.loaded,
            created_at=existing.created_at,
            metadata=existing.metadata_json,
        )

    logger.info("Registering new agent '%s' (template: %s)", template.agent_id, template.template_id)
    agent = RegisteredAgent(
        agent_id=template.agent_id,
        template_id=template.template_id,
        name=template.name,
        description=template.description,
        status="registered",
        loaded=True,
        agent_config=template.model_dump(),
        metadata_json=template.metadata,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    logger.info("Agent '%s' registered successfully", template.agent_id)
    return AgentTemplateResponse(
        agent_id=agent.agent_id,
        template_id=agent.template_id,
        status=agent.status,
        loaded=agent.loaded,
        created_at=agent.created_at,
        metadata=agent.metadata_json,
    )


@router.get("", response_model=list[AgentTemplateResponse])
def list_agents(db: Session = Depends(get_session)) -> list[AgentTemplateResponse]:
    result = db.execute(select(RegisteredAgent).order_by(RegisteredAgent.created_at.desc()))
    agents = result.scalars().all()
    logger.debug("Listed %d agents", len(agents))
    return [
        AgentTemplateResponse(
            agent_id=agent.agent_id,
            template_id=agent.template_id,
            status=agent.status,
            loaded=agent.loaded,
            created_at=agent.created_at,
            metadata=agent.metadata_json,
        )
        for agent in agents
    ]


@router.get("/{agent_id}", response_model=AgentTemplateResponse)
def get_agent(agent_id: str, db: Session = Depends(get_session)) -> AgentTemplateResponse:
    agent = db.get(RegisteredAgent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    logger.debug("Fetched agent '%s'", agent_id)
    return AgentTemplateResponse(
        agent_id=agent.agent_id,
        template_id=agent.template_id,
        status=agent.status,
        loaded=agent.loaded,
        created_at=agent.created_at,
        metadata=agent.metadata_json,
    )