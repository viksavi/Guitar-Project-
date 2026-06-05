import * as THREE from 'three';

export function computeAverageVec(data, vec) {
    var sums = [0, 0, 0]; // x, y, z
    var values = [[], [], []];
    var n = Object.keys(data).length;
    let prop = 5; // proportion
    let nProp = Math.floor(n / prop);

    for (let i = 1; i < n; i++) {
        for (let j = 0; j < 3; j++) {
            values[j].push(data[i][vec][j]);
        }
    }

    for (let j = 0; j < 3; j++) {
        values[j].sort((a, b) => a - b);
    }

    for (let i = nProp; i < nProp * (prop - 1); i++) {
        for (let j = 0; j < 3; j++) {
            sums[j] += values[j][i];
        }
    }

    return [sums[0] / (nProp * (prop - 2)),
            sums[1] / (nProp * (prop - 2)),
            sums[2] / (nProp * (prop - 2))];
}

export function vecDist(vec1, vec2) {
    return Math.sqrt((vec1[0] - vec2[0]) ** 2 + (vec1[1] - vec2[1]) ** 2 + (vec1[2] - vec2[2]) ** 2);
}

export function getSecondsMinutes(t) {
    var seconds = Math.floor(t);
    var minutes = Math.floor(seconds / 60);
    seconds %= 60;
    return { seconds, minutes };    
}

export function formatTime(seconds, minutes) {
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

export function formatPlaybackInfoTime(progress, total) {
    const { seconds: progressSeconds, minutes: progressMinutes } = getSecondsMinutes(progress);
    const { seconds: totalSeconds, minutes: totalMinutes } = getSecondsMinutes(total);
    return `${formatTime(progressSeconds, progressMinutes)} / ${formatTime(totalSeconds, totalMinutes)}`;
}