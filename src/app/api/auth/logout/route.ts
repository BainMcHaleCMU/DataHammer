import { NextResponse } from 'next/server';

export async function POST() {
  // In a real app, you might invalidate the token on the server
  return NextResponse.json({ success: true });
}