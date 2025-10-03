/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#21005D",
        accent: "#5388DF",
        textblue: "#19325C",
        blue: "#19325C",
        white: "#FFFFFF",
      },
    },
  },
  plugins: [],
};
