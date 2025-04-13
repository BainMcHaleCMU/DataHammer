'use client';

import dynamic from 'next/dynamic';
import { Box, VStack, Heading, Button, Text, Textarea, useToast } from '@chakra-ui/react';
import { useState } from 'react';
import { processData } from '@/lib/dataProcessingService';

// Use dynamic import to avoid hydration issues with client components
const FileUpload = dynamic(() => import('@/components/FileUpload'), {
  ssr: false,
});

export default function HomeClient() {
  const toast = useToast();
  const [insights, setInsights] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  const [dataType, setDataType] = useState('csv');

  const handleProcessingInstructions = async () => {
    if (!file) {
      toast({
        title: 'No file selected',
        description: 'Please upload a file first',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (!insights.trim()) {
      toast({
        title: 'No instructions provided',
        description: 'Please enter your data analysis goals',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    
    setIsProcessing(true);
    
    try {
      // Pass the file to the processData function
      const result = await processData(insights, dataType, file);
      
      if (result.success) {
        setAnalysisResult(result.data || {});
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
    } finally {
      setIsProcessing(false);
    }
  };

  // Updated to receive dataType from FileUpload component
  const handleFileSelected = (selectedFile: File, inferredDataType: string) => {
    setFile(selectedFile);
    setDataType(inferredDataType);
  };

  // Add handler for file deletion
  const handleFileDelete = () => {
    setFile(null);
    setDataType('csv');
    setAnalysisResult(null);
  };

  return (
    <Box as="main" minH="100vh" py={8} px={4}>
      <VStack spacing={8} align="stretch" maxW="800px" mx="auto">
        <Heading as="h1" size="xl" textAlign="center">
          DataHammer
        </Heading>
        
        {/* Insights Goals Text Area */}
        <Box w="full">
          <Text mb={2} fontWeight="bold">What insights do you want from your data?</Text>
          <Textarea
            value={insights}
            onChange={(e) => setInsights(e.target.value)}
            placeholder="Example: Predict sales for the next quarter. Help me understand what inventory to stock up on."
            size="lg"
            rows={5}
            resize="vertical"
          />
        </Box>
        
        {/* File Upload Component - Now handles file display and deletion internally */}
        <Box w="full">
          <FileUpload 
            onFileSelect={handleFileSelected} 
            onFileDelete={handleFileDelete}
            selectedFile={file}
          />
        </Box>
        
        {/* Analyze Button */}
        <Button
          colorScheme="blue"
          size="lg"
          onClick={handleProcessingInstructions}
          isLoading={isProcessing}
          loadingText="Analyzing..."
          isDisabled={!file || !insights.trim()}
        >
          Analyze Data
        </Button>
        
        {/* Analysis Results */}
        {analysisResult && (
          <Box w="full" p={6} borderWidth={1} borderRadius="md" bg="white">
            <Heading as="h2" size="md" mb={4}>Analysis Results</Heading>
            
            {/* Report Summary */}
            {analysisResult.summary && (
              <Box mb={6}>
                <Heading as="h3" size="sm" mb={2}>Report Summary</Heading>
                <Box p={3} bg="gray.50" borderRadius="md">
                  {analysisResult.summary.title && (
                    <Text fontWeight="bold">{analysisResult.summary.title}</Text>
                  )}
                  {analysisResult.summary.date && (
                    <Text fontSize="sm" color="gray.600">Generated on: {analysisResult.summary.date}</Text>
                  )}
                </Box>
              </Box>
            )}
            
            {/* Recommendations */}
            {analysisResult.recommendations && analysisResult.recommendations.length > 0 && (
              <Box mb={6}>
                <Heading as="h3" size="sm" mb={2}>Recommendations</Heading>
                <Box p={3} bg="blue.50" borderRadius="md">
                  {Array.isArray(analysisResult.recommendations) ? (
                    <ul>
                      {analysisResult.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  ) : (
                    <Text>{String(analysisResult.recommendations)}</Text>
                  )}
                </Box>
              </Box>
            )}
            
            {/* Visualizations */}
            {analysisResult.visualizations && analysisResult.visualizations.length > 0 && (
              <Box mb={6}>
                <Heading as="h3" size="sm" mb={2}>Visualizations</Heading>
                <Box p={3} bg="green.50" borderRadius="md">
                  <ul>
                    {analysisResult.visualizations.map((viz, index) => (
                      <li key={index}>
                        <Text>
                          <strong>{viz.type}</strong>: {viz.title || viz.data}
                        </Text>
                      </li>
                    ))}
                  </ul>
                </Box>
              </Box>
            )}
            
            {/* Report Content */}
            {analysisResult.content && (
              <Box mb={6}>
                <Heading as="h3" size="sm" mb={2}>Report Content</Heading>
                <Box p={3} bg="purple.50" borderRadius="md">
                  {analysisResult.content.title && (
                    <Text fontWeight="bold" mb={2}>{analysisResult.content.title}</Text>
                  )}
                  
                  {analysisResult.content.sections && (
                    <Box>
                      <Text fontWeight="bold" mb={1}>Sections:</Text>
                      <ul>
                        {analysisResult.content.sections.map((section, index) => (
                          <li key={index}>{section}</li>
                        ))}
                      </ul>
                    </Box>
                  )}
                </Box>
              </Box>
            )}
            
            {/* Raw JSON for debugging */}
            <Box mt={6}>
              <Heading as="h3" size="sm" mb={2}>Raw Data</Heading>
              <Box 
                p={3} 
                bg="gray.100" 
                borderRadius="md" 
                fontSize="sm" 
                fontFamily="monospace"
                overflowX="auto"
              >
                <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
              </Box>
            </Box>
          </Box>
        )}
      </VStack>
    </Box>
  );
}