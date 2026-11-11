// Audio-only ScreenCaptureKit bridge. stdout: interleaved stereo Int16 / 48 kHz.
import Foundation
import ScreenCaptureKit
import AVFoundation

func report(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

@available(macOS 13.0, *)
final class AudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    var stream: SCStream?
    let queue = DispatchQueue(label: "FluentScope.system-audio")

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            throw NSError(domain: "FluentScope", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No display available for system audio capture."])
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48000
        config.channelCount = 2
        let capture = SCStream(filter: filter, configuration: config, delegate: self)
        // No screen output is registered or written.
        try capture.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        stream = capture
        try await capture.startCapture()
        report("READY")
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        report("System audio capture stopped: \(error.localizedDescription)")
        exit(1)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sample: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, sample.isValid,
              let description = sample.formatDescription else { return }
        let format = AVAudioFormat(cmAudioFormatDescription: description)
        guard format.commonFormat == .pcmFormatFloat32,
              format.sampleRate == 48000, format.channelCount == 2,
              let pcm = AVAudioPCMBuffer(pcmFormat: format,
                                        frameCapacity: AVAudioFrameCount(sample.numSamples)) else { return }
        pcm.frameLength = pcm.frameCapacity
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sample, at: 0, frameCount: Int32(sample.numSamples), into: pcm.mutableAudioBufferList)
        guard status == noErr, let channels = pcm.floatChannelData else { return }
        var output = [Int16](repeating: 0, count: sample.numSamples * 2)
        for frame in 0..<sample.numSamples {
            for channel in 0..<2 {
                let value = format.isInterleaved ? channels[0][frame * 2 + channel] : channels[channel][frame]
                output[frame * 2 + channel] = value.isFinite
                    ? Int16(max(-32768, min(32767, value * 32768))) : 0
            }
        }
        output.withUnsafeBytes { FileHandle.standardOutput.write(Data($0)) }
    }
}

if CommandLine.arguments.contains("--check") {
    if #available(macOS 13.0, *) {
        print("ScreenCaptureKit audio available (macOS 13+)")
        exit(0)
    }
    report("System audio capture requires macOS 13 or later.")
    exit(1)
}

if #available(macOS 13.0, *) {
    let capture = AudioCapture()
    Task {
        do {
            try await capture.start()
        } catch {
            report("Cannot capture system audio: \(error.localizedDescription). Allow Screen & System Audio Recording in System Settings > Privacy & Security, then restart FluentScope.")
            exit(1)
        }
    }
    dispatchMain()
} else {
    report("System audio capture requires macOS 13 or later.")
    exit(1)
}
