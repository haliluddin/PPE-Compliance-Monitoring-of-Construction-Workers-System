// frontend/src/context/useAudioUnlock.jsx
import { useState, useEffect } from "react";

export default function useAudioUnlock(audioUrl) {
  const [audio, setAudio] = useState(null);
  const [unlocked, setUnlocked] = useState(false);

  useEffect(() => {
    if (!audioUrl) return;

    const newAudio = new Audio(audioUrl);
    newAudio.preload = "auto";
    newAudio.volume = 1;
    setAudio(newAudio);

    const unlockAudio = () => {
      newAudio.play().catch(() => {});
      setUnlocked(true);
      window.removeEventListener("click", unlockAudio);
      window.removeEventListener("keydown", unlockAudio);
    };

    window.addEventListener("click", unlockAudio);
    window.addEventListener("keydown", unlockAudio);

    return () => {
      window.removeEventListener("click", unlockAudio);
      window.removeEventListener("keydown", unlockAudio);
    };
  }, [audioUrl]);

  const play = () => {
    if (audio && unlocked) audio.play().catch(() => {});
  };

  return play;
}
