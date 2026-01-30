class SentenceBuilder:
    def __init__(self):
        self.sentence = ""       
        self.temp_char = ""      
        self.frame_count = 0     
        
        # CONFIG
        self.THRESHOLD = 15        # ~0.5s to type the FIRST letter
        self.REPEAT_THRESHOLD = 35 # ~1.2s to type the SECOND letter (Double letter)
        # Total hold time for 'LL' = 50 frames (~1.7s)

    def process(self, prediction):
        """
        Returns: (current_sentence, event_triggered)
        """
        event = None 

        # 1. No Hand / lost tracking
        if prediction is None:
            self.temp_char = ""
            self.frame_count = 0
            return self.sentence, event

        # 2. New Character Detected (Reset timer)
        if prediction != self.temp_char:
            self.temp_char = prediction
            self.frame_count = 0 
        
        # 3. Same Character Held (Increment timer)
        else:
            self.frame_count += 1
            
            # --- TRIGGER 1: Initial Add (Normal Typing) ---
            if self.frame_count == self.THRESHOLD:
                event = self._add_input(prediction)
            
            # --- TRIGGER 2: Repeat Add (Double Letter Logic) ---
            elif self.frame_count == (self.THRESHOLD + self.REPEAT_THRESHOLD):
                event = self._add_input(prediction)
                # Reset counter to THRESHOLD so it waits REPEAT_THRESHOLD again for a 3rd letter
                self.frame_count = self.THRESHOLD 
        
        return self.sentence, event

    def _add_input(self, char):
        """Helper to handle space, del, and letters"""
        if char == 'nothing':
            return None
            
        if char == 'space':
            self.sentence += " "
            return "SPEAK" # Trigger audio
            
        if char == 'del':
            self.sentence = self.sentence[:-1]
            return None
            
        # Normal Letter (A, B, C...)
        self.sentence += char
        return None

    def clear(self):
        self.sentence = ""