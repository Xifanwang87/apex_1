import logging
import os
import pickle
import zmq

BROKER_ADDR = 'tcp://10.13.11.250:18055'

#  This is the version of MDP/Client we implement
C_CLIENT = b"APEX.C01"

#  This is the version of MDP/Worker we implement
W_WORKER = b"APEX.W01"

#  MDP/Server commands, as strings
READY         =   b"\001"
REQUEST       =   b"\002"
REPLY         =   b"\003"
HEARTBEAT     =   b"\004"
DISCONNECT    =   b"\005"
RESOURCES     =   b"\006"

COMMANDS = {
    None: b"",
    "READY": READY,
    "REQUEST": REQUEST,
    "REPLY": REPLY,
    "HEARTBEAT": HEARTBEAT,
    "DISCONNECT": DISCONNECT,
    "RESOURCES": RESOURCES
}

HEARTBEAT_LIVENESS = 5 # 3-5 is reasonable
HEARTBEAT_DELAY = 2500 # Heartbeat delay, msecs
RECONNECT_DELAY = 2500 # Reconnect delay, msecs

class ApexServiceClient(object):
    """Majordomo Protocol Client API, Python version.
      Implements the MDP/Worker spec at http:#rfc.zeromq.org/spec:7.
    """
    broker = None
    ctx = None
    client = None
    poller = None
    timeout = 120000
    verbose = False

    def __init__(self, broker, verbose=False):
        self.broker = broker
        self.verbose = verbose
        self.ctx = zmq.Context()
        self.poller = zmq.Poller()
        logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.INFO)
        self.reconnect_to_broker()

    def __del__(self):
        self.poller.unregister(self.client)
        self.client.close()
        self.ctx.term()

    def reconnect_to_broker(self):
        """Connect or reconnect to broker"""
        if self.client:
            self.poller.unregister(self.client)
            self.client.close()
        self.client = self.ctx.socket(zmq.DEALER)
        self.client.linger = 0
        self.client.connect(self.broker)
        self.poller.register(self.client, zmq.POLLIN)
        if self.verbose:
            logging.info("I: connecting to broker at %s...", self.broker)

    def send(self, service, request):
        """Send request to broker
        """
        if not isinstance(request, list):
            request = [request]

        # Prefix request with protocol frames
        # Frame 0: empty (REQ emulation)
        # Frame 1: "MDPCxy" (six bytes, MDP/Client x.y)
        # Frame 2: Service name (printable string)

        request = [b'', C_CLIENT, service] + request
        if self.verbose:
            logging.warn("I: send request to '%s' service: ", service)

        for ix, part in enumerate(request):
            if isinstance(part, str):
                part = part.encode('utf8')
                request[ix] = part
            elif not isinstance(part, bytes):
                request[ix] = pickle.dumps(part)
        self.client.send_multipart(request)

    def recv(self):
        """Returns the reply message or None if there was no reply."""
        try:
            items = self.poller.poll(self.timeout)
        except KeyboardInterrupt:
            return  # interrupted
        if items:
            # if we got a reply, process it
            msg = self.client.recv_multipart()
            if self.verbose:
                logging.info("I: received reply:")

            # Don't try to handle errors, just assert noisily
            assert len(msg) >= 4

            _ = msg.pop(0)
            header = msg.pop(0)
            assert C_CLIENT == header

            _ = msg.pop(0)
            return msg
        else:
            logging.warn("W: permanent error, abandoning request")


    def get_resources(self, service_name):
        request = [b'', C_CLIENT, service_name, RESOURCES]
        if self.verbose:
            logging.warn("I: send request to '%s' service: ", service_name)

        for ix, part in enumerate(request):
            if isinstance(part, str):
                part = part.encode('utf8')
                request[ix] = part
            elif not isinstance(part, bytes):
                request[ix] = pickle.dumps(part)
        self.client.send_multipart(request)
        resources = self.recv()
        return [x.decode('utf8') for x in resources]
