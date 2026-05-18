import asyncio
import json
import logging
import os
import random
from contextlib import suppress
from typing import Callable, Dict, Optional

from dotenv import load_dotenv
from aio_pika import Channel, connect, connect_robust, DeliveryMode, Message
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustConnection
from aiormq.exceptions import AMQPConnectionError, ChannelInvalidStateError, ChannelNotFoundEntity

load_dotenv()


async def create_queue_publish(queue_name, result):
    if not isinstance(queue_name, str) or not queue_name:
        raise ValueError("queue_name must be a non-empty string.")
    if not isinstance(result, dict):
        raise ValueError("result must be a dictionary.")

    host = os.getenv("RABBITMQ_HOST")
    port = int(os.getenv("RABBITMQ_PORT", 5672))
    user = os.getenv("RABBITMQ_USER")
    password = os.getenv("RABBITMQ_PASSWORD")
    heartbeat = int(os.getenv("RABBITMQ_HEARTBEAT", 60))

    if not host or not user or not password:
        raise ValueError("RabbitMQ connection settings are not configured.")

    connection = None
    channel = None
    try:
        connection = await connect(
            host=host,
            port=port,
            login=user,
            password=password,
            heartbeat=heartbeat,
            loop=asyncio.get_running_loop(),
        )
        channel = await connection.channel()
        await channel.declare_queue(
            queue_name,
            durable=True,
            arguments={"x-expires": 1200000}
        )
        await channel.default_exchange.publish(
            Message(
                body=json.dumps(result).encode('utf-8'),
                content_type="application/json",
                delivery_mode=DeliveryMode.PERSISTENT
            ),
            routing_key=queue_name,
        )
        logging.info(f"Delivered and took action: Published to queue '{queue_name}': {result}.")
    finally:
        if channel is not None and not channel.is_closed:
            with suppress(Exception):
                await channel.close()
        if connection is not None and not connection.is_closed:
            with suppress(Exception):
                await connection.close()

class RabbitMQManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.url = os.getenv("RABBITMQ_HOST")
        self.port = int(os.getenv("RABBITMQ_PORT", 5672))
        self.user = os.getenv("RABBITMQ_USER")
        self.password = os.getenv("RABBITMQ_PASSWORD")
        self.queues = [
            "delta_comparator",
            "completion_queue",
            "heartbeat_queue"
        ]
        self.task_queue = "delta_comparator"
        self.completion_queue = "completion_queue"
        # Lower heartbeat for better dead connection detection
        self.heartbeat = int(os.getenv("RABBITMQ_HEARTBEAT", 60))
        self._connection: Optional[AbstractRobustConnection] = None
        self._channels: Dict[str, Channel] = {}
        self._loop = None
        self._lock = asyncio.Lock()
        self._connected = asyncio.Event()
        self._initialized = True

    def _reset_loop_bound_state(self, loop: asyncio.AbstractEventLoop):
        if self._loop is loop:
            return
        if self._loop is not None:
            logging.debug("RabbitMQManager detected event loop change; resetting cached connection and channels.")
        self._loop = loop
        self._connection = None
        self._channels = {}
        self._lock = asyncio.Lock()
        self._connected = asyncio.Event()

    async def _connect(self):
        self._reset_loop_bound_state(asyncio.get_running_loop())
        if self._connected.is_set() and self._connection and not self._connection.is_closed:
            return
        async with self._lock:
            if self._connected.is_set() and self._connection and not self._connection.is_closed:
                return
            self._connected.clear()
            if self._connection:
                with suppress(Exception):
                    await self._connection.close()
                self._connection = None
                self._channels.clear()
            retry_delay = 5
            while not self._connected.is_set():
                try:
                    self._connection = await connect_robust(
                        host=self.url,
                        port=self.port,
                        login=self.user,
                        password=self.password,
                        heartbeat=self.heartbeat,
                        loop=self._loop
                    )
                    self._channels.clear()
                    self._connected.set()
                    logging.info("Connected to RabbitMQ.")
                    return
                except (AMQPConnectionError, OSError, ChannelNotFoundEntity) as e:
                    logging.warning(f"RabbitMQ connection failed: {e}. Retrying in {retry_delay}s.")
                    if self._connection:
                        with suppress(Exception):
                            await self._connection.close()
                        self._connection = None
                except Exception as e:
                    logging.critical(f"Unexpected RabbitMQ connection error: {e}")
                    if self._connection:
                        with suppress(Exception):
                            await self._connection.close()
                        self._connection = None
                await asyncio.sleep(retry_delay + (retry_delay * 0.1 * random.random()))  # Add jitter
                retry_delay = min(retry_delay * 2, 60)

    async def get_channel(self, name: str) -> Channel:
        self._reset_loop_bound_state(asyncio.get_running_loop())
        await self._connect()
        channel = self._channels.get(name)
        # Check both channel and connection state
        if channel:
            # If channel is closed or connection is dead, remove from cache
            if channel.is_closed or not self._connection or self._connection.is_closed:
                self._channels.pop(name, None)
                channel = None
        if not channel:
            # Defensive: check connection state again before creating channel
            if not self._connection or self._connection.is_closed:
                self._channels.pop(name, None)
                raise ConnectionError("No active RabbitMQ connection.")
            try:
                channel = await self._connection.channel()
                # Declare queues only once per channel
                for q in self.queues:
                    args = {"x-message-ttl": 1000} if q == "heartbeat_queue" else None
                    await channel.declare_queue(q, durable=True, arguments=args)
                self._channels[name] = channel
            except (AMQPConnectionError, ChannelInvalidStateError) as e:
                # If we get ChannelInvalidStateError, the connection is likely dead. Force reconnect.
                self._connected.clear()
                if self._connection and not self._connection.is_closed:
                    with suppress(Exception):
                        await self._connection.close()
                self._connection = None
                self._channels.pop(name, None)
                logging.error(f"[get_channel] Channel/connection error: {e}. Forcing reconnect.")
                raise ConnectionError(f"Failed to establish channel: {e}") from e
            except Exception as e:
                self._channels.pop(name, None)
                logging.error(f"Error creating channel: {e}", exc_info=True)
                raise
        # Final check: if channel is still not open, force reconnect next time
        if channel.is_closed:
            self._channels.pop(name, None)
            self._connected.clear()
            raise ConnectionError("Channel is closed after creation attempt.")
        return channel

    async def publish_message(self, routing_key: str, body: dict, correlation_id: Optional[str] = None):
        message_body = json.dumps(body).encode('utf-8')
        msg_id = correlation_id or body.get("task_id", "N/A")
        for attempt in range(1, 4):
            channel = None
            try:
                channel = await self.get_channel("publish")
                await channel.default_exchange.publish(
                    Message(
                        body=message_body,
                        content_type="application/json",
                        delivery_mode=DeliveryMode.PERSISTENT,
                        correlation_id=correlation_id
                    ),
                    routing_key=routing_key,
                )
                logging.debug(f"[{msg_id}] Published to '{routing_key}'.")
                return
            except (ChannelInvalidStateError, AMQPConnectionError, ConnectionError, AttributeError, TimeoutError) as e:
                logging.warning(f"[{msg_id}] Publish attempt {attempt} failed: {e}")
                if channel:
                    with suppress(Exception):
                        await channel.close()
                self._channels.pop("publish", None)
                self._connected.clear()
                # On error, force reconnect for next attempt
                if attempt == 3:
                    logging.error(f"[{msg_id}] Failed to publish after 3 attempts.", exc_info=True)
                    raise
                await asyncio.sleep(attempt)
            except asyncio.CancelledError:
                logging.warning(f"[{msg_id}] Publish cancelled.")
                return
            except Exception as e:
                self._channels.pop("publish", None)
                logging.error(f"[{msg_id}] Unexpected publish error: {e}", exc_info=True)
                raise

    async def publish_completion(self, task_id: str, completion_data: dict):
        await self.publish_message(
            routing_key=self.completion_queue,
            body={"task_id": task_id, **completion_data},
            correlation_id=task_id
        )

    async def publish_heartbeat(self, heartbeat_data: dict):
        try:
            await self.publish_message(routing_key="heartbeat_queue", body=heartbeat_data)
        except Exception as e:
            logging.error(f"Failed to publish heartbeat: {e}")

    async def start_consuming(self, callback: Callable[[dict], asyncio.Future]):
        async def _process_message(message: AbstractIncomingMessage):
            task_id = "unknown_task"
            try:
                await message.ack()
                logging.debug(f"[{task_id}] Message acknowledged.")
                data = json.loads(message.body.decode())
                task_id = data.get("task_id", f"delivery_tag_{message.delivery_tag}")
                logging.debug(f"[{task_id}] Received task.")
                await callback(data)
            except json.JSONDecodeError:
                logging.error(f"Failed to decode message: {message.body[:200]}. Rejecting.")
                with suppress(Exception):
                    await message.nack(requeue=False)
            except asyncio.CancelledError:
                logging.warning(f"[{task_id}] Task cancelled.")
                with suppress(Exception):
                    await message.nack(requeue=False)
            except Exception as e:
                logging.error(f"[{task_id}] Error processing message: {e}", exc_info=True)
                failure_data = {"status": "failed", "result": {"error": f"Processing error: {e}"}}
                try:
                    await self.publish_completion(task_id, failure_data)
                except Exception as pub_e:
                    logging.error(f"[{task_id}] Failed to publish failure status: {pub_e}", exc_info=True)

        # Robust consumer: restart on connection/channel errors
        while True:
            try:
                channel = await self.get_channel("consume")
                await channel.set_qos(prefetch_count=int(os.getenv("RABBITMQ_PREFETCH_COUNT", 1)))
                queue = await channel.get_queue(self.task_queue)
                await queue.consume(_process_message)
                logging.info(f"Consumer started on '{self.task_queue}'.")
                break  # Exit loop if started successfully
            except (AMQPConnectionError, ConnectionError, ChannelInvalidStateError) as e:
                logging.error(f"Cannot start consuming: {e}. Retrying in 5s.")
                self._channels.pop("consume", None)
                self._connected.clear()
                await asyncio.sleep(5)
            except Exception as e:
                logging.critical(f"Failed to start consumer: {e}", exc_info=True)
                raise

    async def stop_consuming(self):
        # await asyncio.gather(*(channel.close() for channel in self._channels.values()), return_exceptions=True)
        self._channels.clear()

    async def close(self):
        await self.stop_consuming()
        if self._connection and not self._connection.is_closed:
            with suppress(Exception):
                await self._connection.close()
        self._connected.clear()

    async def create_queue_publish(self, queue_name, result):
        if not isinstance(queue_name, str) or not queue_name:
            raise ValueError("queue_name must be a non-empty string.")
        if not isinstance(result, dict):
            raise ValueError("result must be a dictionary.")
        channel = await self.get_channel("publish")
        try:
            await channel.declare_queue(
                queue_name,
                durable=True,
                arguments={"x-expires": 1200000}
            )
            await channel.default_exchange.publish(
                Message(
                    body=json.dumps(result).encode('utf-8'),
                    content_type="application/json",
                    delivery_mode=DeliveryMode.PERSISTENT
                ),
                routing_key=queue_name,
            )
            logging.info(f"Delivered and took action: Published to queue '{queue_name}': {result}.")
        except Exception as e:
            logging.error(f"Failed to publish to queue '{queue_name}': {e}", exc_info=True)