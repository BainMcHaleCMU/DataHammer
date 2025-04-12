/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  // Add basePath for GitHub Pages deployment
  // Comment this out for local development
  // basePath: '/your-repo-name',
};

module.exports = nextConfig;