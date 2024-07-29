from locust import HttpUser, task, between, FastHttpUser

class EchoUser(FastHttpUser):
    # wait_time = between(0, 0.1)  # Simulate a wait time between requests
    host = "http://127.0.0.1:4200"

    def echo(self):
        self.client.post("/echo", json={"input_integer": 42})

    @task
    def get42(self):
        self.client.get("/42")