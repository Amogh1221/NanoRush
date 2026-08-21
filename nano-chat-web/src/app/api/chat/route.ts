import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  const { messages } = await req.json();

  // Forward to our FastAPI backend running on port 8000
  const response = await fetch('http://127.0.0.1:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  });

  if (!response.ok) {
    return NextResponse.json({ error: 'Backend failed' }, { status: 500 });
  }

  // Create a TransformStream to pass the SSE chunks directly to the client
  const { readable, writable } = new TransformStream();
  
  // Pipe the backend stream to the Next.js stream
  response.body?.pipeTo(writable);

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
