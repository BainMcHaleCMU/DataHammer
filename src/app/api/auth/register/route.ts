import { NextRequest, NextResponse } from 'next/server';

// Mock user database - in a real app, this would be a database
let users = [
  {
    id: '1',
    email: 'user@example.com',
    password: 'password123', // In a real app, this would be hashed
    displayName: 'Demo User',
  },
];

export async function POST(request: NextRequest) {
  try {
    const { email, password, displayName } = await request.json();

    // Check if user already exists
    if (users.some(u => u.email === email)) {
      return NextResponse.json(
        { error: 'Email already in use' },
        { status: 400 }
      );
    }

    // Create new user
    const newUser = {
      id: `${users.length + 1}`,
      email,
      password, // In a real app, this would be hashed
      displayName: displayName || email.split('@')[0],
    };

    users.push(newUser);

    // Create a simple token (in a real app, use JWT)
    const token = Buffer.from(`${newUser.id}:${Date.now()}`).toString('base64');

    // Return user data and token
    return NextResponse.json({
      user: {
        id: newUser.id,
        email: newUser.email,
        displayName: newUser.displayName,
      },
      token,
    });
  } catch (error) {
    console.error('Registration error:', error);
    return NextResponse.json(
      { error: 'Registration failed' },
      { status: 500 }
    );
  }
}