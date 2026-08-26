// Decode a local hand recording with Chromium and write a timeline contact sheet.

import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import http from "node:http";
import { createRequire } from "node:module";
import path from "node:path";


const require = createRequire(import.meta.url);
const { chromium } = require("playwright");


const projectRoot = path.resolve(import.meta.dirname, "..");
const sourcePath = path.resolve(
  process.argv[2] ?? path.join(projectRoot, "video", "85e7c0fa7ef1e973fd0a8d605e11b41a_raw.mp4"),
);
const outputPath = path.resolve(
  process.argv[3] ?? path.join(projectRoot, "video", "hand-video-timeline.png"),
);
const requestedStartSec = Number(process.argv[4]);
const requestedEndSec = Number(process.argv[5]);
const requestedSampleCount = Number(process.argv[6]);

async function launchBrowser() {
  for (const channel of ["chrome", "msedge", undefined]) {
    try {
      return await chromium.launch({
        channel,
        headless: true,
        args: ["--allow-file-access-from-files", "--autoplay-policy=no-user-gesture-required"],
      });
    } catch (error) {
      if (channel === undefined) throw error;
    }
  }
  throw new Error("No Chromium-compatible browser is available");
}

async function serveVideo(filePath) {
  const stat = await fs.stat(filePath);
  const server = http.createServer((request, response) => {
    if (request.url !== "/video.mp4") {
      response.writeHead(404).end();
      return;
    }
    const range = request.headers.range;
    if (!range) {
      response.writeHead(200, {
        "Accept-Ranges": "bytes",
        "Access-Control-Allow-Origin": "*",
        "Content-Length": stat.size,
        "Content-Type": "video/mp4",
      });
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
      "Accept-Ranges": "bytes",
      "Access-Control-Allow-Origin": "*",
      "Content-Length": end - start + 1,
      "Content-Range": `bytes ${start}-${end}/${stat.size}`,
      "Content-Type": "video/mp4",
    });
    createReadStream(filePath, { start, end }).pipe(response);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  return { server, url: `http://127.0.0.1:${address.port}/video.mp4` };
}

const videoServer = await serveVideo(sourcePath);
const browser = await launchBrowser();
try {
  const page = await browser.newPage();
  const sourceUrl = videoServer.url;
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
      video.addEventListener("error", () => reject(new Error(JSON.stringify({
        code: video.error?.code,
        message: video.error?.message,
        networkState: video.networkState,
        readyState: video.readyState,
        src: video.currentSrc,
      }))), { once: true });
      video.load();
    });
    return {
      duration: video.duration,
      width: video.videoWidth,
      height: video.videoHeight,
    };
  }, sourceUrl);

  const marginSec = Math.min(0.1, metadata.duration / 100);
  const startSec = Number.isFinite(requestedStartSec) ? requestedStartSec : marginSec;
  const endSec = Number.isFinite(requestedEndSec) ? requestedEndSec : metadata.duration - marginSec;
  const sampleCount = Number.isInteger(requestedSampleCount) && requestedSampleCount >= 2
    ? requestedSampleCount
    : 20;
  const times = Array.from({ length: sampleCount }, (_, index) => (
    startSec + (endSec - startSec) * index / (sampleCount - 1)
  ));
  const sheetDataUrl = await page.evaluate(async ({ times, metadata }) => {
    const video = document.querySelector("#source-video");
    const columns = 4;
    const thumbWidth = 320;
    const thumbHeight = Math.round(thumbWidth * metadata.height / metadata.width);
    const labelHeight = 28;
    const rows = Math.ceil(times.length / columns);
    const canvas = document.createElement("canvas");
    canvas.width = columns * thumbWidth;
    canvas.height = rows * (thumbHeight + labelHeight);
    const context = canvas.getContext("2d");
    context.fillStyle = "#111827";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.font = "18px Arial";
    context.textBaseline = "middle";

    async function seek(time) {
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error(`Seek timeout at ${time}`)), 10000);
        video.addEventListener("seeked", () => {
          clearTimeout(timeout);
          resolve();
        }, { once: true });
        video.currentTime = time;
      });
    }

    for (let index = 0; index < times.length; index += 1) {
      await seek(times[index]);
      const column = index % columns;
      const row = Math.floor(index / columns);
      const x = column * thumbWidth;
      const y = row * (thumbHeight + labelHeight);
      context.drawImage(video, x, y, thumbWidth, thumbHeight);
      context.fillStyle = "#111827";
      context.fillRect(x, y + thumbHeight, thumbWidth, labelHeight);
      context.fillStyle = "#f8fafc";
      context.fillText(`${times[index].toFixed(2)} s`, x + 8, y + thumbHeight + labelHeight / 2);
    }
    return canvas.toDataURL("image/png");
  }, { times, metadata });

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, Buffer.from(sheetDataUrl.split(",", 2)[1], "base64"));
  process.stdout.write(`${JSON.stringify({
    sourcePath,
    outputPath,
    ...metadata,
    sampledStartSec: startSec,
    sampledEndSec: endSec,
    sampleCount,
  }, null, 2)}\n`);
} finally {
  await browser.close();
  await new Promise((resolve) => videoServer.server.close(resolve));
}
