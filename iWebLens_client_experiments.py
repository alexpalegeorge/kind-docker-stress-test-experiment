# Generate the parallel requests based on the ThreadPool Executor
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import sys
import time
import glob
import requests
import threading
import uuid
import base64
import json
import os


# send http request
def call_object_detection_service(image):
    try:

        url = str(sys.argv[2])
        data = {}
        # generate uuid for image
        id = uuid.uuid5(uuid.NAMESPACE_OID, image)
        # Encode image into base64 string
        with open(image, 'rb') as image_file:
            data['image'] = base64.b64encode(image_file.read()).decode('utf-8')

        data['id'] = str(id)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=json.dumps(data), headers=headers)

        if response.ok:
            output = "Thread : {},  input image: {},  output:{}".format(threading.current_thread().getName(),
                                                                        image, response.text)
            print(output)
        else:
            print("Error, response status:{}".format(response))

    except Exception as e:
        print("Exception in webservice call: {}".format(e))


# gets list of all images path from the input folder
def get_images_to_be_processed(input_folder):
    images = []
    for image_file in glob.iglob(input_folder + "*.jpg"):
        images.append(image_file)
    return images


def main(num_threads):
    ## provide arguments-> input folder, url, number of workers

    if len(sys.argv) != 3:
        raise ValueError("Arguments list is wrong. Please use the following format: {} {} {}".
                         format("python iWebLens_client.py", "<input_folder>", "<URL>"))  # "<number_of_workers>"))

    input_folder = os.path.join(sys.argv[1], "")
    images = get_images_to_be_processed(input_folder)
    num_images = images.__len__()

    num_workers = num_threads

    start_time = time.time()
    # create a worker  thread  to  invoke the requests in parallel
    with PoolExecutor(max_workers=num_workers) as executor:
        for _ in executor.map(call_object_detection_service, images):
            pass
    elapsed_time = time.time() - start_time
    print("Total time spent: {} average response time: {}".format(elapsed_time, elapsed_time / num_images))

    return elapsed_time / num_images


# New function for reporting mean response time (3 passes) on a variety of thread counts
def experiment():
    thread_tests = [1, 6, 11, 16, 21, 26, 31]
    output = {}
    for threads in thread_tests:
        round = 0
        times = []
        while round < 3:
            times.append(main(threads))
            print("finished round", round + 1, "with time", times[round])
            round += 1
        output[str(threads), "thread"] = (times[0] + times[1] + times[2]) / 3
    print(output)


if __name__ == "__main__":
    experiment()
