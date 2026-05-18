"""This module contains the configuration settings for the requirement comparison tracer."""
from typing import Union
from pydantic import Field
from pydantic_settings import BaseSettings


class BaseConf(BaseSettings):  # pylint: disable=too-few-public-methods
    """Default parameters for all environments."""

    API_PREFIX: str = "/api/v1"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(
        default=True,
        description="To have the app run in development mode, for features such as hot "
        "reload and others",
    )
    PROJECT_NAME: str = Field(
        default="Requirement Comparision tracer Service",
        description="Signifies the project name. Shows up in the doc page",
    )


config = dict(local=BaseConf)
settings: Union[BaseConf] = config.get("local")()
