import AppKit

// OpenCV's Cocoa waitKey sleeps between polls. Keep its existing NSWindow,
// dispatch pending AppKit events without imposing a sleep on every video frame.
@_cdecl("sonar_poll_key")
public func sonarPollKey() -> Int32 {
    precondition(Thread.isMainThread)
    return MainActor.assumeIsolated {
        var key: Int32 = -1
        let application = NSApplication.shared
        for _ in 0..<128 {
            guard let event = application.nextEvent(matching: .any,
                until: Date.distantPast, inMode: .default, dequeue: true) else { break }
            if event.type == .keyDown, let value = event.charactersIgnoringModifiers?.utf8.first {
                key = Int32(value)
            }
            application.sendEvent(event)
        }
        application.updateWindows()
        return key
    }
}

// Prepare offline accessibility audio before the video clock starts.
// The same main-thread helper owns playback; no speech synthesis during playback.
import AVFoundation
private var sonarAudio: AVAudioPlayer?
@_cdecl("sonar_audio_prepare")
public func sonarAudioPrepare(_ path: UnsafePointer<CChar>?) -> Int32 {
    precondition(Thread.isMainThread)
    guard let path = path else { return 0 }
    do {
        sonarAudio = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: String(cString: path)))
        return sonarAudio?.prepareToPlay() == true ? 1 : 0
    } catch { sonarAudio = nil; return 0 }
}
@_cdecl("sonar_audio_play")
public func sonarAudioPlay() -> Int32 { return sonarAudio?.play() == true ? 1 : 0 }
@_cdecl("sonar_audio_pause")
public func sonarAudioPause() { sonarAudio?.pause() }
@_cdecl("sonar_audio_stop")
public func sonarAudioStop() { sonarAudio?.stop(); sonarAudio?.currentTime = 0 }
@_cdecl("sonar_audio_playing")
public func sonarAudioPlaying() -> Int32 { return sonarAudio?.isPlaying == true ? 1 : 0 }
