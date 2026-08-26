// Extract the selected two-second recording segment as square PNG frames.

import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import http from "node:http";
import { createRequire } from "node:module";
import path from "node:path";


const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const projectRoot = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(
  projectRoot,
  "video",
  "85e7c0fa7ef1e973fd0a8d605e11b41a_raw.mp4",
);
const outputDirectory = path.join(projectRoot, "tmp", "recorded-hand-frames");

const START_SEC = 4.75;
const DURATION_SEC = 2.0;
const OUTPUT_FPS = 20;
const FRAME_COUNT = DURATION_SEC * OUTPUT_FPS;
const OUTPUT_SIZE = 512;


async function serveVideo(filePath) {
  const stat = await fs.stat(filePath);
  const server = http.createServer((request, response) => {
    if (request.url !== "/video.mp4") {
      response.writeHead(404).end();
      return;
    }
    const headers = {
      "Accept-Ranges": "bytes",
      "Access-Control-Allow-Origin": "*",
      "Content-Type": "video/mp4",
    };
    const range = request.headers.range;
    if (!range) {
      response.writeHead(200, { ...headers, "Content-Length": stat.size });
      createReadStream(filePath).pipe(response);
      return;
    }
    const match = /^bytes=(\d+)-(\d*)$/.exec(range);
    if (!match) {
      response.writeHead(416).end();
      return;
    }
    const start = Number(match[1]);
    const end = match[2] ? Number(match[2]) : stat.size - 1;
    response.writeHead(206, {
      ...headers,
      "Content-Length": end - start + 1,
      "Content-Range": `bytes ${start}-${end}/${stat.size}`,
    });
    createReadStream(filePath, { start, end }).pipe(response);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  return { server, url: `http://127.0.0.1:${address.port}/video.mp4` };
}


async function launchBrowser() {
  for (const channel of ["chrome", "msedge", undefined]) {
    try {
      return await chromium.launch({ channel, headless: true });
    } catch (error) {
      if (channel === undefined) throw error;
    }
  }
  throw new Error("No Chromium-compatible browser is available");
}


const videoServer = await serveVideo(sourcePath);
const browser = await launchBrowser();
try {
  const page = await browser.newPage();
  const metadata = await page.evaluate(async (src) => {
    const video = document.createElement("video");
    video.id = "source-video";
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.preload = "auto";
    video.src = src;
    document.body.append(video);
    await new Promise((resolve, reject) => {
      video.addEventListener("loadedmetadata", resolve, { once: true });
      video.addEventListener("error", () => reject(new Error(video.error?.message)), { once: true });
      video.load();
    });
    return { width: video.videoWidth, height: video.videoHeight, duration: video.duration };
  }, videoServer.url);
  if (START_SEC + DURATION_SEC > metadata.duration) {
    throw new Error("Selected segment exceeds the recording duration");
  }

  const cropSize = metadata.width;
  const cropX = 0;
  // Leave a little more room above the fully extended fingertips while keeping
  // the wrist exit outside the lower edge of the square cue.
  const cropY = Math.round((metadata.height - cropSize) * 0.24);
  await fs.mkdir(outputDirectory, { recursive: true });
  for (let index = 0; index < FRAME_COUNT; index += 1) {
    const timestamp = START_SEC + index / OUTPUT_FPS;
    const dataUrl = await page.evaluate(async ({ timestamp, cropX, cropY, cropSize, outputSize }) => {
      const video = document.querySelector("#source-video");
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error(`Seek timeout at ${timestamp}`)), 10000);
        video.addEventListener("seeked", () => {
          clearTimeout(timeout);
          resolve();
        }, { once: true });
        video.currentTime = timestamp;
      });
      const canvas = document.createElement("canvas");
      canvas.width = outputSize;
      canvas.height = outputSize;
      const context = canvas.getContext("2d");
      context.drawImage(
        video,
        cropX,
        cropY,
        cropSize,
        cropSize,
        0,
        0,
        outputSize,
        outputSize,
      );
      return canvas.toDataURL("image/png");
    }, { timestamp, cropX, cropY, cropSize, outputSize: OUTPUT_SIZE });
    const framePath = path.join(outputDirectory, `raw-${String(index).padStart(3, "0")}.png`);
    await fs.writeFile(framePath, Buffer.from(dataUrl.split(",", 2)[1], "base64"));
  }
  await fs.writeFile(
    path.join(outputDirectory, "metadata.json"),
    `${JSON.stringify({
      sourcePath,
      startSec: START_SEC,
      durationSec: DURATION_SEC,
      fps: OUTPUT_FPS,
      frameCount: FRAME_COUNT,
      outputSize: OUTPUT_SIZE,
      crop: { x: cropX, y: cropY, size: cropSize },
      source: metadata,
    }, null, 2)}\n`,
  );
  process.stdout.write(`extracted ${FRAME_COUNT} frames to ${outputDirectory}\n`);
} finally {
  await browser.close();
  await new Promise((resolve) => videoServer.server.close(resolve));
}
