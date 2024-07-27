from locust import HttpUser, task, between

class EchoUser(HttpUser):
    wait_time = between(0, 0.1)  # Simulate a wait time between requests
    host = "http://127.0.0.1:4200"

    @task
    def echo(self):
        self.client.post("/echo", json={"input_integer": 42})