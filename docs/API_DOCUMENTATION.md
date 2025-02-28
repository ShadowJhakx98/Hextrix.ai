# API Documentation

## Overview
This document outlines the API endpoints, input/output formats, and use cases for the system's modules. It is intended to help developers integrate and extend the system seamlessly.

---

## API Endpoints

### 1. `/query`
- **Description**: Processes user queries and returns AI-generated responses.
- **Method**: POST
- **Input**:
  ```json
  {
    "query": "string"
  }
  ```
- **Output**:
  ```json
  {
    "response": "string"
  }
  ```
- **Example**:
  ```bash
  curl -X POST http://localhost:5000/query -H "Content-Type: application/json" -d '{"query": "Tell me a joke."}'
  ```

### 2. `/capture_screen`
- **Description**: Captures a screenshot of the current screen.
- **Method**: GET
- **Output**:
  - PNG file of the captured screen.

### 3. `/ocr`
- **Description**: Performs OCR on an uploaded image.
- **Method**: POST
- **Input**:
  - Multipart form-data containing the image file.
- **Output**:
  ```json
  {
    "text": "string"
  }
  ```
- **Example**:
  ```bash
  curl -X POST http://localhost:5000/ocr -F "file=@image.png"
  ```

### 4. `/emotion`
- **Description**: Analyzes the sentiment and returns the detected emotion.
- **Method**: POST
- **Input**:
  ```json
  {
    "text": "string"
  }
  ```
- **Output**:
  ```json
  {
    "emotion": "string"
  }
  ```

### 5. `/tts`
- **Description**: Converts text to speech.
- **Method**: POST
- **Input**:
  ```json
  {
    "text": "string"
  }
  ```
- **Output**:
  - Audio file of the generated speech.

### 6. `/stt`
- **Description**: Converts speech to text.
- **Method**: POST
- **Input**:
  - Audio file in WAV or MP3 format.
- **Output**:
  ```json
  {
    "text": "string"
  }
  ```

### 7. `/lighting`
- **Description**: Updates RGB lighting based on emotional state.
- **Method**: POST
- **Input**:
  ```json
  {
    "emotion": "string"
  }
  ```
- **Output**:
  ```json
  {
    "status": "success"
  }
  ```

### 8. `/discord_bot/command`
- **Description**: Sends a command to the Discord bot.
- **Method**: POST
- **Input**:
  ```json
  {
    "command": "string"
  }
  ```
- **Output**:
  ```json
  {
    "response": "string"
  }
  ```

---

## Input and Output Formats

### General Input Structure
All JSON inputs should adhere to the following format:
```json
{
  "key": "value"
}
```

### General Output Structure
API responses will generally follow this format:
```json
{
  "status": "success",
  "data": { ... }
}
```

---

## Use Cases

### Querying the AI
- **Scenario**: A user wants to get an answer or perform a task via the AI.
- **Steps**:
  1. Send a POST request to `/query` with the user query.
  2. Receive the AI response in the output JSON.

### Screen Capture
- **Scenario**: Capture and process the current screen for analysis.
- **Steps**:
  1. Send a GET request to `/capture_screen`.
  2. Save the returned PNG file locally.

### Emotional Sentiment Lighting
- **Scenario**: Adjust RGB lighting to match an emotional tone.
- **Steps**:
  1. Send a POST request to `/lighting` with the detected emotion.
  2. Receive a success confirmation.

---

## Diagrams and Workflow

### Example: AI Query Workflow
```plaintext
User -> [POST: /query] -> Flask App -> AI Module -> Response
```

### Example: OCR Processing
```plaintext
Image Upload -> [POST: /ocr] -> OCR Module -> Extracted Text
```

---

## Developer Notes
- Ensure proper error handling and validation for all inputs.
- Test endpoints thoroughly for both expected and edge cases.
- Document any new endpoints or parameters in this file.

By adhering to these guidelines, developers can effectively utilize and extend the system's APIs.

