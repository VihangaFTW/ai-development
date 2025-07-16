# JS and React Revision Plan for Graduate Dev Role

This document serves as a comprehensive revision plan to prepare for a JavaScript and React interview. It covers essential concepts, with code examples to reinforce learning.

---

## 1. JavaScript Fundamentals

### 1.1 Variables and Data Types
- var, let, const
- Primitive types (string, number, boolean, null, undefined, symbol)
- Objects and Arrays

**Example:**
```javascript
const name = 'Alice';
let age = 25;
const isDeveloper = true;
const person = { name: 'Alice', age: 25 };
const colors = ['red', 'blue', 'green'];
```

### 1.2 Functions
- Function declarations and expressions
- Arrow functions
- Callbacks
- Immediately Invoked Function Expressions (IIFE)

**Example:**
```javascript
function greet(name) {
  return `Hello, ${name}`;
}

const add = (a, b) => a + b;

setTimeout(() => console.log('Delayed message'), 1000);
```

### 1.3 Scope and Closures
- Block scope with let/const
- Function scope with var
- Closure concept

**Example:**
```javascript
function outer() {
  let counter = 0;
  return function inner() {
    counter++;
    return counter;
  };
}
const increment = outer();
console.log(increment()); // 1
```

### 1.4 Asynchronous JavaScript
- Callbacks
- Promises
- async/await

**Example:**
```javascript
const fetchData = () => {
  return new Promise((resolve) => {
    setTimeout(() => resolve('Data fetched'), 1000);
  });
};

async function getData() {
  const data = await fetchData();
  console.log(data);
}
getData();
```

---

## 2. JavaScript ES6+ Features

### 2.1 Destructuring
**Example:**
```javascript
const { name, age } = person;
const [firstColor] = colors;
```

### 2.2 Modules
**Example:**
```javascript
// export
export default function sum(a, b) { return a + b; }

// import
import sum from './sum.js';
```

### 2.3 Spread and Rest Operators
**Example:**
```javascript
const newArray = [...colors, 'yellow'];
const newObj = { ...person, hobby: 'reading' };
```

---

## 3. React Fundamentals

### 3.1 Components
- Functional components
- Class components
- Props and State

**Example:**
```jsx
function Greeting({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// Using State with hooks
import React, { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}
```

### 3.2 Lifecycle Methods (Class Components)
- componentDidMount
- componentDidUpdate
- componentWillUnmount

### 3.3 Hooks (Functional Components)
- useState
- useEffect
- useContext

### 3.4 Event Handling
**Example:**
```jsx
<button onClick={() => alert('Clicked!')}>Click me</button>
```

### 3.5 Conditional Rendering
**Example:**
```jsx
{isLoggedIn ? <Dashboard /> : <LoginForm />}
```

---

## 4. React Routing and State Management

- React Router basics
- Context API
- Redux (brief overview)

---

## 5. TypeScript

### 5.1 Basics of TypeScript
- Type annotations
- Interfaces and Types
- Enums

**Example:**
```typescript
interface Person {
  name: string;
  age: number;
}

const person: Person = { name: 'Alice', age: 25 };
```

### 5.2 TypeScript with React
- Typing props and state
- Using generics

**Example:**
```tsx
interface GreetingProps {
  name: string;
}

function Greeting({ name }: GreetingProps) {
  return <h1>Hello, {name}!</h1>;
}

// State typing
const [count, setCount] = useState<number>(0);
```

### 5.3 Configuring TypeScript in Projects
- tsconfig.json basics
- Compilation and transpilation

---