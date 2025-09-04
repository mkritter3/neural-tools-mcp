#!/usr/bin/env node

// Check Node version
const nodeVersion = process.version;
const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);

if (majorVersion < 18) {
  console.error('❌ Neural SDK requires Node.js 18 or higher');
  console.error(`   Current version: ${nodeVersion}`);
  console.error('   Please upgrade Node.js from: https://nodejs.org');
  process.exit(1);
}

// Import and run CLI
require('../dist/cli.js');