import { NextRequest, NextResponse } from 'next/server';
import { unlink } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';

export async function DELETE(request: NextRequest) {
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
    
    // Delete file
    await unlink(filePath);
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('File deletion error:', error);
    return NextResponse.json(
      { error: 'Failed to delete file' },
      { status: 500 }
    );
  }
}