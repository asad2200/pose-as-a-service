from locust import FastHttpUser, TaskSet, task, between, events
from itertools import cycle
import base64, uuid, os

IMAGE_DIR = os.getenv("IMAGE_DIR", "./experiments/inputfolder")
JSON_ENDPOINT = os.getenv("JSON_ENDPOINT", "/pose/json")
ANNOTATION_ENDPOINT = os.getenv("ANNOT_ENDPOINT", "/pose/image")

# Load and cache images **once per worker process**, not per user
def _load_images(dir_: str):
    imgs = []
    for f in os.listdir(dir_):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            with open(os.path.join(dir_, f), "rb") as fp:
                imgs.append(base64.b64encode(fp.read()).decode())
    if not imgs:
        raise RuntimeError(f"No images found in {dir_}")
    return imgs

IMAGES = _load_images(IMAGE_DIR)
IMAGE_CYCLE = cycle(IMAGES)        # thread‑safe iterator

# Locust tasks
class PoseTasks(TaskSet):
    def _next_image(self):
        return next(IMAGE_CYCLE)

    @task(2)
    def json_inference(self):
        with self.client.post(
            JSON_ENDPOINT,
            json={"id": str(uuid.uuid4()), "image": self._next_image()},
            name="JSON_Inference",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Status {response.status_code}")
        
    @task(1)
    def annotation_inference(self):
        with self.client.post(
            ANNOTATION_ENDPOINT,
            json={"id": str(uuid.uuid4()), "image": self._next_image()},
            name="Annotation_Inference",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Status {response.status_code}")

class PoseUser(FastHttpUser):      # FastHttpUser ≈ 30‑40 % less CPU
    tasks = [PoseTasks]
    wait_time = between(1, 3)
    user_id_counter = 0  # give every user an index so we know who failed
    
    def on_start(self):
        self.user_id = PoseUser.user_id_counter
        PoseUser.user_id_counter += 1

# ----------  GLOBAL HOOK: stop on first failure  ----------
TEST_ENV = None                     # we’ll save the Environment object here

# Grab the Environment object when the test starts
@events.test_start.add_listener
def _(environment, **kw):
    global TEST_ENV
    TEST_ENV = environment

# Catch every request (success + failure) but act only on the first failure
@events.request.add_listener
def on_any_request(request_type, name, response_time, response_length,
                   response, context, exception, start_time, url, **kw):
    global TEST_ENV

    # ignore successes and anything after the first failure
    if exception is None:
        return

    # --------- what you wanted to capture ----------
    users_now = TEST_ENV.runner.user_count
    avg_ms    = TEST_ENV.stats.total.avg_response_time
    # -----------------------------------------------

    print(
        f"First failure! active_users={users_now}, "
        f"overall_avg={avg_ms:.2f} ms (up to this point)"
    )

    # stop the entire run
    TEST_ENV.runner.quit()


# locust \
#   -f experiments/locustfile.py \
#   --headless \
#   --users 2000 \
#   --spawn-rate 1 \
#   --host http://207.211.146.117:30000


# locust \
#   -f locustfile.py \
#   --headless \
#   --users 2000 \
#   --spawn-rate 1 \
#   --only-summary \
#   --host http://localhost:30000