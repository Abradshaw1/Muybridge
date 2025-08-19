//
//  ModelBenchTests.swift
//  Muybridge
//
//  Created by Aidan Bradshaw on 07.08.2025.
//


import XCTest
import CoreML
import os

final class ModelBenchTests: XCTestCase {

    // ======= EDIT THESE =======
    private let MODEL_NAME = "UNetStep256"             // base name only (no extension)
    //private let TENSOR_INPUT_NAME = "rgb_norm"              // your model's tensor input key
    private let TENSOR_INPUT_NAMES = ["rgb_latent", "target_latent"] // for unet only
    private let LAYOUT: Layout = .nchw                   // .nchw or .nhwc
    private let TENSOR_DTYPE: MLMultiArrayDataType = .float16 // or .float16 if your model expects it
    private let COMPUTE_UNITS: MLComputeUnits = .all // try .cpuOnly / .cpuAndGPU too
    private let WIDTH = 32, HEIGHT = 32                // fixed dummy input size
    private let WARMUP_ITERS = 5, MEASURE_ITERS = 100  // benchmark knobs
    // ==========================

    enum Layout { case nchw, nhwc }

    private let log = OSLog(subsystem: "bench.app", category: .pointsOfInterest)

    // MARK: - Model loading (.mlmodelc / .mlpackage / .mlmodel)
    private func loadModel() throws -> MLModel {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = COMPUTE_UNITS
        let b = Bundle(for: Self.self)

        if let url = b.url(forResource: MODEL_NAME, withExtension: "mlmodelc") {
            return try MLModel(contentsOf: url, configuration: cfg)
        }
        if let pkg = b.url(forResource: MODEL_NAME, withExtension: "mlpackage") {
            return try MLModel(contentsOf: pkg, configuration: cfg)
        }
        if let src = b.url(forResource: MODEL_NAME, withExtension: "mlmodel") {
            let compiled = try MLModel.compileModel(at: src)
            return try MLModel(contentsOf: compiled, configuration: cfg)
        }
        fatalError("Couldn’t find \(MODEL_NAME).mlmodelc/.mlpackage/.mlmodel in bundle. Check target membership (Copy Bundle Resources).")
    }

    private func modelSizeBytes() -> Int? {
        let b = Bundle.main
        for ext in ["mlmodelc", "mlpackage", "mlmodel"] {
            if let url = b.url(forResource: MODEL_NAME, withExtension: ext),
               let vals = try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey]) {
                return vals.totalFileAllocatedSize
            }
        }
        return nil
    }

    // MARK: - Dummy tensor input (zero-filled)
    // for unet input benchmark
    private func makeTensorFeature() -> MLFeatureProvider {
        // pick shape NCHW vs NHWC
        let shape: [NSNumber] = {
            switch LAYOUT {
            case .nchw: return [1, 4, NSNumber(value: HEIGHT), NSNumber(value: WIDTH)]
            case .nhwc: return [1, NSNumber(value: HEIGHT), NSNumber(value: WIDTH), 4]
            }
        }()
        
        var dict: [String: MLFeatureValue] = [:]
        for name in TENSOR_INPUT_NAMES {
            let arr = try! MLMultiArray(shape: shape, dataType: TENSOR_DTYPE)
            let bytesPerElem = (TENSOR_DTYPE == .float16 ? 2 : 4)
            memset(arr.dataPointer, 0, arr.count * bytesPerElem)
            dict[name] = MLFeatureValue(multiArray: arr)
        }
        let tArr = try! MLMultiArray(shape: [1], dataType: .float32)
            tArr[0] = 0.0 // or any valid timestep in your inference schedule
            dict["t_f32"] = MLFeatureValue(multiArray: tArr)
        
        
        
        return try! MLDictionaryFeatureProvider(dictionary: dict)
    }
    
    
//    private func makeTensorFeature() -> MLFeatureProvider {
//        let shape: [NSNumber]
//        switch LAYOUT {
//        case .nchw: shape = [1, 3, NSNumber(value: HEIGHT), NSNumber(value: WIDTH)] // [N,C,H,W]
//        case .nhwc: shape = [1, NSNumber(value: HEIGHT), NSNumber(value: WIDTH), 3] // [N,H,W, C]
//        }
//        let arr = try! MLMultiArray(shape: shape, dataType: TENSOR_DTYPE)
//        let elemSize = (TENSOR_DTYPE == .float16) ? 2 : 4
//        memset(arr.dataPointer, 0, arr.count * elemSize)
//        return try! MLDictionaryFeatureProvider(dictionary: [TENSOR_INPUT_NAME: MLFeatureValue(multiArray: arr)])
//    }

    private func percentile(_ xs: [Double], _ p: Double) -> Double {
        let s = xs.sorted()
        let i = Int(Double(s.count - 1) * p)
        return s[i]
    }

    // Optional: print model IO once to confirm input key/shape/dtype
    private func printModelIO(_ model: MLModel) {
        print("=== Model Inputs ===")
        for (k, v) in model.modelDescription.inputDescriptionsByName {
            print("• \(k): \(v.type) shape=\(v.multiArrayConstraint?.shape ?? []) dtype=\(String(describing: v.multiArrayConstraint?.dataType))")
        }
        print("=== Model Outputs ===")
        for (k, v) in model.modelDescription.outputDescriptionsByName { print("• \(k): \(v.type)") }
    }
    
    private func currentMemoryFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<natural_t>.size)

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }

        guard result == KERN_SUCCESS else { return 0 }
        return info.phys_footprint // in bytes
    }

    // ---------------------------------------------------------------------------
    // Generic tensor-only benchmark that prints the *same* stats block you
    // already like.
    // ---------------------------------------------------------------------------
    private func runBenchmark(
        modelName:      String,
        makeInput:      @autoclosure () -> MLFeatureProvider,
        inputDesc:      String,
        layout:         Layout,
        dataType:       MLMultiArrayDataType,
        width:          Int,
        height:         Int,
        warmup:         Int = 5,
        measure:        Int = 100,
        units:          MLComputeUnits = .all
    ) throws {
        // -- load ----------------------------------------------------------------
        let cfg = MLModelConfiguration(); cfg.computeUnits = units
        let url = Bundle(for: Self.self)
                 .url(forResource: modelName, withExtension: "mlmodelc")!
        let model = try MLModel(contentsOf: url, configuration: cfg)

        // -- mem before ----------------------------------------------------------
        let memBefore = currentMemoryFootprint()
        os_log("Memory before inference: %.2f MB", Double(memBefore) / 1_048_576.0)

        // -- warm-up -------------------------------------------------------------
        for _ in 0..<warmup { _ = try model.prediction(from: makeInput()) }

        // -- timed loop ----------------------------------------------------------
        var times: [Double] = []
        var peak = memBefore
        for i in 0..<measure {
            let sid = OSSignpostID(UInt64(i))
            os_signpost(.begin, log: log, name: "inference", signpostID: sid)
            let t0 = DispatchTime.now().uptimeNanoseconds
            _ = try model.prediction(from: makeInput())
            let t1 = DispatchTime.now().uptimeNanoseconds
            os_signpost(.end, log: log, name: "inference", signpostID: sid)
            times.append(Double(t1 - t0) / 1e6)
            peak = max(peak, currentMemoryFootprint())
        }

        // -- stats block identical to the old test -------------------------------
        let mean = times.reduce(0,+) / Double(times.count)
        let p50  = percentile(times, 0.50)
        let p90  = percentile(times, 0.90)
        let p99  = percentile(times, 0.99)
        let fps  = 1000.0 / mean
        let memAfter = currentMemoryFootprint()
        let memDelta = memAfter > memBefore ? memAfter - memBefore : 0
        let avgPowerW: Double? = 1.5                           // Instruments › Energy Log
        let energy_uJ: String = {
            guard let w = avgPowerW else { return "" }
            let J = w * (mean / 1000.0)                        // mean is ms → s
            return String(format: "%.0f µJ", J * 1_000_000.0)
        }()
        
        let csvURL = FileManager.default.urls(for: .documentDirectory,
                                              in: .userDomainMask)[0]
                       .appendingPathComponent("\(modelName)_latencies_ms.csv")
        try? (["ms"] + times.map { String(format:"%.4f",$0) })
             .joined(separator: "\n")
             .write(to: csvURL, atomically: true, encoding: .utf8)

        print("""
        === Core ML Benchmark (tensor-only: \(modelName)) ===
        Input tensor: \(inputDesc), dtype: \(dataType)
        Units: \(units) | Iterations: \(measure) (after \(warmup) warm-up)
        Latency (ms): p50=\(String(format:"%.2f",p50)) | \
        p90=\(String(format:"%.2f",p90)) | p99=\(String(format:"%.2f",p99)) | \
        mean=\(String(format:"%.2f",mean))
        FPS / Throughput: \(String(format:"%.1f",fps)) inf/s
        Model size (bytes): \(modelSizeBytes() ?? -1)
        Energy per inference: \(energy_uJ)
        Memory footprint: before = \(String(format:"%.2f",Double(memBefore)/1_048_576)) MB \
        after = \(String(format:"%.2f",Double(memAfter)/1_048_576)) MB \
        delta = \(String(format:"%.2f",Double(memDelta)/1_048_576)) MB \
        peak = \(String(format:"%.2f",Double(peak)/1_048_576)) MB
        Peak NPU (ANE) RAM: (hidden by Core ML)
        ===========================================
        """)
    }
    
    

    // MARK: - Benchmark
    func test_benchmark_tensor_only() throws {
        let model = try loadModel()
        printModelIO(model) // helpful first run; keep or remove later
        
        let memBefore = currentMemoryFootprint()
        os_log("Memory before inference: %.2f MB", Double(memBefore) / 1_048_576.0)

        // Warm-up
        for _ in 0..<WARMUP_ITERS { _ = try model.prediction(from: makeTensorFeature()) }

        // Timed loop
        var ms: [Double] = []; ms.reserveCapacity(MEASURE_ITERS)
        var peakMemoryBytes: UInt64 = memBefore
        
        for i in 0..<MEASURE_ITERS {
            let inp = makeTensorFeature()
            let sid = OSSignpostID(UInt64(i))
            os_signpost(.begin, log: log, name: "inference", signpostID: sid)
            let t0 = DispatchTime.now().uptimeNanoseconds
            _ = try model.prediction(from: inp)
            let t1 = DispatchTime.now().uptimeNanoseconds
            os_signpost(.end, log: log, name: "inference", signpostID: sid)
            ms.append(Double(t1 - t0) / 1_000_000.0) // ms
            
            let current = currentMemoryFootprint()
            if current > peakMemoryBytes {
                peakMemoryBytes = current
                
            }
        }

        // Stats
        let mean = ms.reduce(0,+) / Double(ms.count)
        let p50 = percentile(ms, 0.50), p90 = percentile(ms, 0.90), p99 = percentile(ms, 0.99)
        let fps = 1000.0 / mean
        let throughput = fps // batch=1

        // CSV
        let csv = (["ms"] + ms.map { String(format: "%.4f", $0) }).joined(separator: "\n")
        let doc = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let csvURL = doc.appendingPathComponent("\(MODEL_NAME)_latencies_ms.csv")
        try? csv.write(to: csvURL, atomically: true, encoding: .utf8)

        // Model size
        let sizeBytes = modelSizeBytes() ?? -1
        

        // Energy-per-inference placeholder (paste Avg Power W from Instruments)
        let avgPowerW: Double? = 1.5 // e.g., 1.8
        let energy_uJ: String = {
            guard let w = avgPowerW else { return "N/A (set avgPowerW)" }
            let J = w * (mean / 1000.0)
            return String(format: "%.0f", J * 1_000_000.0)
        }()
        
        
        let memAfter = currentMemoryFootprint()
        let memDelta = memAfter > memBefore ? memAfter - memBefore : 0
        let memPeak = peakMemoryBytes

        os_log("Memory after inference: %.2f MB", Double(memAfter) / 1_048_576.0)
        os_log("Memory delta: %.2f MB", Double(memDelta) / 1_048_576.0)
        os_log("Peak memory during inference: %.2f MB", Double(memPeak) / 1_048_576.0)


        print("""
        === Core ML Benchmark (tensor-only: \(MODEL_NAME)) ===
        Input tensor: \(LAYOUT == .nchw ? "[1,3,\(HEIGHT),\(WIDTH)] NCHW" : "[1,\(HEIGHT),\(WIDTH),3] NHWC"), dtype: \(TENSOR_DTYPE)
        Units: \(COMPUTE_UNITS)
        Iterations: \(MEASURE_ITERS) (after \(WARMUP_ITERS) warm-up)
        Latency (ms): p50=\(String(format:"%.2f", p50)) | p90=\(String(format:"%.2f", p90)) | p99=\(String(format:"%.2f", p99)) | mean=\(String(format:"%.2f", mean))
        FPS: \(String(format:"%.1f", fps)) | Throughput (inf/s): \(String(format:"%.1f", throughput))
        Model size (bytes): \(sizeBytes)
        Energy per inference (µJ): \(energy_uJ)
        Latency CSV: \(csvURL.path)
        NOTE: Peak RAM + Average Power via Instruments (Memory / Energy Log). ANE (NPU) RAM isn’t exposed.
        ===========================================
        """)
    }
    
    // MARK: - === ElifPoseModel === ---------------------------------------------
    private func makePoseFeature() -> MLFeatureProvider {
        let arr = try! MLMultiArray(
            shape: [1, 3, 256, 192] as [NSNumber],
            dataType: .float32)
        memset(arr.dataPointer, 0, arr.count * 4)   // 4 bytes/Float32
        return try! MLDictionaryFeatureProvider(
            dictionary: ["input": MLFeatureValue(multiArray: arr)])
    }

    func test_benchmark_elifpose_tensor() throws {
        try runBenchmark(
            modelName: "ElifPoseModel",
            makeInput: makePoseFeature(),
            inputDesc: "[1,3,256,192] NCHW",
            layout: .nchw,
            dataType: .float32,
            width: 192,
            height: 256)
    }
    
    
    // MARK: - === ElifDetModel (640×640 pixel-buffer) ===
    private func makeDetFeature() -> MLFeatureProvider {
        // 1. create an all-black BGRA pixel-buffer (640 × 640)
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 640, 640,
                            kCVPixelFormatType_32BGRA,
                            nil, &pb)

        // 2. wrap it in the feature dictionary that matches the model’s input keys
        return try! MLDictionaryFeatureProvider(dictionary: [
            "image"              : MLFeatureValue(pixelBuffer: pb!),
            "iouThreshold"       : MLFeatureValue(double: 0.45),   // optional
            "confidenceThreshold": MLFeatureValue(double: 0.25)    // optional
        ])
    }

    func test_benchmark_elifdet_pixel() throws {
        try runBenchmark(
            modelName: "ElifDetModel",           // compiled .mlmodelc name
            makeInput: makeDetFeature(),         // <- image feature
            inputDesc: "CVPixelBuffer 640×640",  // just for the banner
            layout:    .nchw,                    // dummy value; not used for images
            dataType:  .float32,                 // ditto
            width:     640,
            height:    640)
    }
    
    
    
}






