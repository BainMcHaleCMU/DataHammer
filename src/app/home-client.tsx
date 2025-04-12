'use client';

import dynamic from 'next/dynamic';
import { Box } from '@chakra-ui/react';

// Use dynamic import to avoid hydration issues with client components
const FileUpload = dynamic(() => import('@/components/FileUpload'), {
  ssr: false,
});

export default function HomeClient() {
  return (
    <Box as="main" minH="100vh" py={8} px={4}>
      <FileUpload />
    </Box>
  );
}