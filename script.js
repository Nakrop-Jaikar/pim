const video = document.getElementById('video');

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadModel() {
    const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet);
    return detector;
}

async function detectPose(detector) {
    const poses = await detector.estimatePoses(video);
    // ตรวจสอบท่าทางที่นี่
    console.log(poses);
    requestAnimationFrame(() => detectPose(detector));
}

async function main() {
    await setupCamera();
    video.play();
    const detector = await loadModel();
    detectPose(detector);
}

main();