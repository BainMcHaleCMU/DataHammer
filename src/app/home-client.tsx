'use client';

import dynamic from 'next/dynamic';
import { Box, VStack, Heading, Divider, useToast } from '@chakra-ui/react';
import { processData } from '@/lib/dataProcessingService';

// Use dynamic import to avoid hydration issues with client components
const FileUpload = dynamic(() => import('@/components/FileUpload'), {
  ssr: false,
});

const UserInputForm = dynamic(() => import('@/components/UserInputForm'), {
  ssr: false,
});

export default function HomeClient() {
  const toast = useToast();

  const handleProcessingInstructions = async (instructions: string, dataType: string) => {
    try {
      const result = await processData(instructions, dataType);
      
      if (result.success) {
        toast({
          title: 'Success',
          description: result.message,
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Error',
          description: result.error || 'An error occurred',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to process data',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <Box as="main" minH="100vh" py={8} px={4}>
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="xl" textAlign="center">
          DataHammer
        </Heading>
        
        <FileUpload />
        
        <Divider my={4} />
        
        <UserInputForm onSubmit={handleProcessingInstructions} />
      </VStack>
    </Box>
  );
}