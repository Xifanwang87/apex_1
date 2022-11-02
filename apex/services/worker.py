import zmq
import logging
import time
import zmq


def dump(msg_or_socket):
    """Receives all message parts from socket, printing each frame neatly"""
    if isinstance(msg_or_socket, zmq.Socket):
        # it's a socket, call on current message
        msg = msg_or_socket.recv_multipart()
    else:
        msg = msg_or_socket
    print("----------------------------------------")
    for part in msg:
        print("[%03d]" % len(part), end=' ')
        is_text = True
        print(part)

class ApexServiceWorker(object):
    """ApexServiceWorker is the base class for all of our services."""
    SERVICE_NAME = None
    RESOURCES = None
    def __init__(self, broker, verbose=False):
        self.service = self.SERVICE_NAME
        self.broker = broker
        self.verbose = verbose
        self.ctx = zmq.Context()
        self.poller = zmq.Poller()
        logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.INFO)
        self.worker = None

        self.heartbeat_at = 0 # When to send HEARTBEAT (relative to time.time(), so in seconds)
        self.liveness = 0 # How many attempts left

        # Internal state
        self.expect_reply = False # False only at start
        self.timeout = 2400 # poller timeout

        # Return address, if any
        self.reply_to = None

        self.reconnect_to_broker()
        self.loop()

    def reconnect_to_broker(self):
        """Connect or reconnect to broker"""
        if self.worker:
            self.poller.unregister(self.worker)
            self.worker.close()
        self.worker = self.ctx.socket(zmq.DEALER)
        self.worker.linger = 0
        self.worker.connect(self.broker)
        self.poller.register(self.worker, zmq.POLLIN)
        if self.verbose:
            logging.info("I: connecting to broker at %s...", self.broker)

        # Register service with broker
        self.send(MDP.READY, self.service, [])

        # If liveness hits zero, queue is considered disconnected
        self.liveness = MDP.HEARTBEAT_LIVENESS
        self.heartbeat_at = time.time() + 1e-3 * MDP.HEARTBEAT_DELAY

    def send(self, command, option=None, msg=None):
        """
        Send message to broker.
        If no msg is provided, creates one internally
        """
        if msg is None:
            msg = []
        elif not isinstance(msg, list):
            msg = [msg]

        if option:
            msg = [option] + msg

        msg = [b'', MDP.W_WORKER, command] + msg
        if self.verbose:
            logging.info("I: sending %s to broker", command)
            dump(msg)

        for ix, part in enumerate(msg):
            if isinstance(part, str):
                part = part.encode('utf8')
                msg[ix] = part
            elif not isinstance(part, bytes):
                msg[ix] = pickle.dumps(part)
        self.worker.send_multipart(msg)

    def _on_message(self, message):
        if len(message) == 1:
            if message[0] == MDP.RESOURCES:
                if isinstance(self.RESOURCES, dict):
                    return list(self.RESOURCES.keys())
                else:
                    return list(self.RESOURCES)
        return self.on_message(message)

    def on_message(self, message):
        raise NotImplementedError("Need to implement on_message")

    def loop(self):
        """Send reply, if any, to broker and wait for next request."""
        # Format and send the reply if we were provided one
        reply = None

        while True:
            # Poll socket for a reply, with timeout
            try:
                items = self.poller.poll(self.timeout)
            except KeyboardInterrupt:
                break # Interrupted

            if items:
                msg = self.worker.recv_multipart()
                if self.verbose:
                    logging.info("I: received message from broker: ")
                    dump(msg)

                self.liveness = MDP.HEARTBEAT_LIVENESS
                # Don't try to handle errors, just assert noisily
                assert len(msg) >= 3

                empty = msg.pop(0)
                assert empty == b''

                header = msg.pop(0)
                assert header == MDP.W_WORKER

                command = msg.pop(0)
                if command == MDP.REQUEST:
                    self.reply_to = msg.pop(0)
                    # pop empty
                    empty = msg.pop(0)
                    assert empty == b''
                    reply = self._on_message(msg) # We have a request to process

                elif command == MDP.HEARTBEAT:
                    # Do nothing for heartbeats
                    pass
                elif command == MDP.DISCONNECT:
                    self.reconnect_to_broker()
                elif command == MDP.RESOURCES:
                    reply = list(self.RESOURCES.keys())
                else :
                    logging.error("E: invalid input message: ")
                    dump(msg)
            else:
                self.liveness -= 1
                if self.liveness == 0:
                    if self.verbose:
                        logging.warn("W: disconnected from broker - retrying...")
                    try:
                        time.sleep(1e-3*MDP.RECONNECT_DELAY)
                    except KeyboardInterrupt:
                        break
                    self.reconnect_to_broker()

            # Send HEARTBEAT if it's time
            if time.time() > self.heartbeat_at:
                self.send(MDP.HEARTBEAT)
                self.heartbeat_at = time.time() + 1e-3*MDP.HEARTBEAT_DELAY

            if reply is not None:
                assert self.reply_to is not None
                reply = [self.reply_to, b''] + reply
                self.send(MDP.REPLY, msg=reply)
                reply = None

        logging.warn("W: interrupt received, killing worker...")
        return None

    def destroy(self):
        # context.destroy depends on pyzmq >= 2.1.10
        self.ctx.destroy(0)
