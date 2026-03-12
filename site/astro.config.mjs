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
            head: [
                {
                    tag: "meta",
                    attrs: {
                        name: "author",
                        content: "Emil M. Constantinescu"
                    }
                },
                {
                    tag: "meta",
                    attrs: {
                        name: "citation_author",
                        content: "Emil M. Constantinescu"
                    }
                }
            ],
            social: [
                {
                    icon: "github",
                    label: "GitHub",
                    href: "https://github.com/emconsta/desolve"
                },
                {
                    icon: "external",
                    label: "Author website",
                    href: "https://emconsta.github.io/"
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
