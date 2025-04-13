import { NextRequest, NextResponse } from 'next/server';
import { existsSync } from 'fs';
import { join } from 'path';

export async function GET(request: NextRequest) {
  try {
    const path = request.nextUrl.searchParams.get('path');
    
    if (!path) {
      return NextResponse.json(
        { error: 'No path provided' },
        { status: 400 }
      );
    }
    
    // In a real app, you would validate the path and check permissions
    
    // Check if file exists
    const filePath = join(process.cwd(), 'public', path);
    if (!existsSync(filePath)) {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      );
    }
    
    // Return URL
    const url = path.startsWith('/') ? path : `/${path}`;
    
    return NextResponse.json({ url });
  } catch (error) {
    console.error('File URL error:', error);
    return NextResponse.json(
      { error: 'Failed to get file URL' },
      { status: 500 }
    );
  }
}