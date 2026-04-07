"""Agent and Tool registry with scope-based discovery."""

import logging
from typing import Dict, List, Optional, Type, TypeVar, Set
import importlib.metadata

logger = logging.getLogger(__name__)
T = TypeVar("T")


class AgentRegistry:
    """Agent registry with scope-based discovery and entry_points support."""
    
    _agents: Dict[str, Type] = {}
    _agent_scopes: Dict[str, Set[str]] = {}
    
    @classmethod
    def register(cls, agent_class: Type[T], name: Optional[str] = None, scopes: Optional[List[str]] = None) -> Type[T]:
        agent_name = name or agent_class.__name__
        cls._agents[agent_name] = agent_class
        cls._agent_scopes[agent_name] = set(scopes) if scopes else set()
        return agent_class
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        cls._agents.pop(name, None)
        cls._agent_scopes.pop(name, None)
        return name not in cls._agents
    
    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        return cls._agents.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs):
        agent_class = cls._agents.get(name)
        if not agent_class:
            available = ", ".join(cls._agents.keys()) or "None"
            raise ValueError(f"Agent '{name}' not registered. Available: {available}")
        return agent_class(**kwargs)
    
    @classmethod
    def list(cls, scope: Optional[str] = None) -> List[str]:
        if scope is None:
            return list(cls._agents.keys())
        return [name for name, scopes in cls._agent_scopes.items() 
                if not scopes or scope in scopes]
    
    @classmethod
    def discover(cls) -> int:
        """Discover agents via entry_points 'vllm_omni_cli.agents'."""
        count = 0
        try:
            eps = importlib.metadata.entry_points(group="vllm_omni_cli.agents")
        except AttributeError:
            return 0
        for ep in eps:
            try:
                cls_ = ep.load()
                if isinstance(cls_, type) and hasattr(cls_, '__init__'):
                    cls.register(cls_, name=ep.name)
                    count += 1
            except Exception:
                continue
        return count
    
    @classmethod
    def clear(cls) -> None:
        cls._agents.clear()
        cls._agent_scopes.clear()


def register_agent(cls_or_name=None, scopes: Optional[List[str]] = None):
    """Agent registration decorator.
    
    Usage:
        @register_agent
        class MyAgent(BaseAgent): ...
        
        @register_agent("custom-name", scopes=["hpc"])
        class MyAgent(BaseAgent): ...
    """
    if cls_or_name is None or isinstance(cls_or_name, str):
        name = cls_or_name if isinstance(cls_or_name, str) else None
        def decorator(cls: Type[T]) -> Type[T]:
            return AgentRegistry.register(cls, name=name, scopes=scopes)
        return decorator
    else:
        return AgentRegistry.register(cls_or_name, scopes=scopes)
