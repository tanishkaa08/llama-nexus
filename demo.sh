# This demo script requires `jq` for parsing JSON.
# Please install it with: sudo apt-get install jq  OR  brew install jq
BASE_URL="http://127.0.0.1:3389"
echo

# Step 1: First Request (Start a new conversation)
# I am NOT sending an X-Session-ID header.
# The server should create a new session.
echo "ME(user)-My favorite color is blue. Please remember that."

# I used curl to send the request and store the entire JSON response
RESPONSE1=$(curl -s -X POST "$BASE_URL/responses" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "My favorite color is blue."}')

# Used jq to extract the session_id and the assistant's message
# If this fails, it means the server's response was not valid JSON.
SESSION_ID=$(echo "$RESPONSE1" | jq -r '.session_id')
ASSISTANT_MSG1=$(echo "$RESPONSE1" | jq -r '.choices[0].message.content')

echo "$ASSISTANT_MSG1"
echo "System - Server assigned new Session ID: $SESSION_ID"
echo
echo "--------------------------------------------------"
echo

# Step 2: Second Request (Continue the conversation) 
# Now, we send the SAME session ID back to the server in the header.
echo "ME(user) - What is my favorite color?"

RESPONSE2=$(curl -s -X POST "$BASE_URL/responses" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: $SESSION_ID" \
  -d '{"prompt": "What is my favorite color?"}')

ASSISTANT_MSG2=$(echo "$RESPONSE2" | jq -r '.choices[0].message.content')

echo "$ASSISTANT_MSG2"
echo
echo "DEMO COMPLETE"