'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Box,
  Button,
  Center,
  Flex,
  Heading,
  Icon,
  Text,
  VStack,
  useToast
} from '@chakra-ui/react'
import axios from 'axios'

export default function FileUpload() {
  const [file, setFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const toast = useToast()

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0])
      toast({
        title: 'File uploaded',
        description: `${acceptedFiles[0].name} is ready for analysis`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
    }
  }, [toast])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    maxFiles: 1
  })

  const handleAnalyze = async () => {
    if (!file) {
      toast({
        title: 'No file selected',
        description: 'Please upload a file first',
        status: 'error',
        duration: 3000,
        isClosable: true,
      })
      return
    }

    setIsLoading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setAnalysisResult(response.data)
      toast({
        title: 'Analysis complete',
        description: 'Your data has been analyzed successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
    } catch (error) {
      console.error('Error analyzing file:', error)
      toast({
        title: 'Analysis failed',
        description: 'There was an error analyzing your file',
        status: 'error',
        duration: 3000,
        isClosable: true,
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <VStack spacing={6} w="full" maxW="800px" mx="auto" py={10}>
      <Heading as="h1" size="xl">Data Analytics Dashboard</Heading>
      
      <Box
        {...getRootProps()}
        w="full"
        p={10}
        borderWidth={2}
        borderRadius="md"
        borderStyle="dashed"
        borderColor={isDragActive ? "blue.400" : "gray.300"}
        bg={isDragActive ? "blue.50" : "gray.50"}
        cursor="pointer"
        transition="all 0.2s"
        _hover={{ borderColor: "blue.300", bg: "blue.50" }}
      >
        <input {...getInputProps()} />
        <Center flexDir="column">
          <Icon boxSize={12} color="gray.400" />
          {isDragActive ? (
            <Text mt={4} textAlign="center">Drop the file here...</Text>
          ) : (
            <Text mt={4} textAlign="center">
              Drag and drop a spreadsheet file here, or click to select a file
            </Text>
          )}
          <Text fontSize="sm" color="gray.500" mt={2}>
            Supported formats: .csv, .xls, .xlsx
          </Text>
        </Center>
      </Box>

      {file && (
        <Box w="full" p={4} borderWidth={1} borderRadius="md">
          <Text fontWeight="bold">Selected file:</Text>
          <Text>{file.name} ({(file.size / 1024).toFixed(2)} KB)</Text>
        </Box>
      )}

      <Button
        colorScheme="blue"
        size="lg"
        onClick={handleAnalyze}
        isLoading={isLoading}
        loadingText="Analyzing..."
        isDisabled={!file}
      >
        Analyze Data
      </Button>

      {analysisResult && (
        <Box w="full" p={6} borderWidth={1} borderRadius="md" bg="white">
          <Heading as="h2" size="md" mb={4}>Analysis Results</Heading>
          <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
        </Box>
      )}
    </VStack>
  )
}