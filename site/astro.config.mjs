import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
    site: "https://emconsta.github.io",
    base: "/desolve",
    trailingSlash: "always",
    integrations: [
        starlight({
            title: "DESolve",
            description: "Research-oriented time integration methods for ODEs and semi-discrete PDEs.",
            disable404Route: true,
            favicon: "/favicon.svg",
            social: [
                {
                    icon: "github",
                    label: "GitHub",
                    href: "https://github.com/emconsta/desolve"
                }
            ],
            customCss: ["./src/styles/custom.css"],
            editLink: {
                baseUrl: "https://github.com/emconsta/desolve/edit/main/site/"
            },
            sidebar: [
                {
                    label: "Start Here",
                    items: [
                        { slug: "guides/getting-started" },
                        { slug: "guides/solver-workflow" }
                    ]
                },
                {
                    label: "Methods",
                    autogenerate: { directory: "methods" }
                },
                {
                    label: "Examples",
                    autogenerate: { directory: "examples" }
                },
                {
                    label: "Reference",
                    autogenerate: { directory: "reference" }
                }
            ]
        })
    ]
});
