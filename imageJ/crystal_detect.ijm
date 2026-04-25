// crystal_detect.ijm
// ------------------
// ImageJ/Fiji macro for automated crystal detection on fixed-target chip images.
//
// What it does:
//   1. Converts image to 8-bit grayscale
//   2. Subtracts uneven background (rolling ball, radius=300px)
//   3. Enhances local contrast with CLAHE to bring out low-contrast crystals
//   4. Thresholds to binary mask
//   5. Watershed to separate touching/overlapping crystals
//   6. Analyzes particles, keeps top N by area
//   7. Overlays detected crystals (white fill) on original color image and saves
//
// Usage:
//   Run in Fiji (ImageJ2). You will be prompted for:
//     - Input folder  (preprocessed chip images)
//     - Output folder (annotated results saved as .tif)
//     - How many largest crystals to keep per image
//
// Notes:
//   - Minimum particle size is hardcoded at 300 px^2 — adjust if your magnification differs
//   - CLAHE blocksize=127 and max slope=3 worked well for 20x magnification LaB6 chips
//   - Watershed helps split overlapping crystals but is not perfect for heavily clustered regions

setBatchMode(true);

inputDir  = getDirectory("Choose input folder (preprocessed images)");
outputDir = getDirectory("Choose output folder (results)");
keepCount = getNumber("How many largest crystals to keep per image?", 30);

list = getFileList(inputDir);
run("ROI Manager...");
roiManager("reset");

for (f = 0; f < list.length; f++) {
    filename = list[f];
    nameLower = toLowerCase(filename);

    if (!endsWith(nameLower, ".tif") && !endsWith(nameLower, ".jpg") && !endsWith(nameLower, ".png"))
        continue;

    open(inputDir + filename);
    originalTitle = getTitle();

    run("Duplicate...", "title=Processing_Copy");
    selectWindow("Processing_Copy");

    // --- Preprocessing ---
    run("8-bit");
    run("Subtract Background...", "rolling=300 light");
    run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None* fast_(less_accurate)");

    // --- Binary segmentation ---
    run("Make Binary");

    // --- Watershed: separates touching crystals ---
    run("Watershed");

    // --- Particle detection ---
    run("Set Measurements...", "area redirect=None decimal=3");
    roiManager("reset");
    run("Analyze Particles...", "size=300-Infinity add");

    n = nResults;

    if (n > 0) {
        // Collect and sort areas descending (bubble sort — ImageJ macro has no built-in sort)
        area = newArray(n);
        for (i = 0; i < n; i++) area[i] = getResult("Area", i);

        for (i = 0; i < n - 1; i++) {
            for (j = i + 1; j < n; j++) {
                if (area[j] > area[i]) {
                    temp   = area[i];
                    area[i] = area[j];
                    area[j] = temp;
                }
            }
        }

        // Keep top N (or all if fewer than N detected)
        keepNow  = minOf(n, keepCount);
        minArea  = area[keepNow - 1];

        run("Select None");
        run("Clear Results");
        for (i = 0; i < keepNow; i++) setResult("Area", i, area[i]);
        updateResults();

        // Re-detect with area threshold for top N
        roiManager("reset");
        run("Analyze Particles...", "size=" + minArea + "-Infinity add");

        // Overlay white fills on original image
        selectWindow(originalTitle);
        setForegroundColor(255, 255, 255);
        roiManager("fill");
        run("Flatten");
        saveAs("Tiff", outputDir + "detected_" + filename);

    } else {
        // No crystals found — save original unchanged
        selectWindow(originalTitle);
        run("Flatten");
        saveAs("Tiff", outputDir + "detected_" + filename);
        print("No crystals detected: " + filename);
    }

    close("*");
}

// Cleanup
if (isOpen("ROI Manager")) { selectWindow("ROI Manager"); run("Close"); }
if (isOpen("Results"))     { selectWindow("Results");     run("Close"); }
if (isOpen("Summary"))     { selectWindow("Summary");     run("Close"); }

setBatchMode(false);
print("Done. Results saved to: " + outputDir);
