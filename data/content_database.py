"""
Content Database for Meaningful Recommendations
Contains realistic social media content with categories, titles, and descriptions
"""

import random

class ContentDatabase:
    def __init__(self):
        self.content_items = {
            # Technology & Programming Content
            1: {
                'title': 'Python Machine Learning Tutorial',
                'description': 'Complete guide to building ML models with Python and scikit-learn',
                'category': 'Technology',
                'content_type': 'Educational Video',
                'tags': ['python', 'machine-learning', 'tutorial'],
                'creator': 'TechGuru123'
            },
            2: {
                'title': 'React.js Best Practices 2024',
                'description': 'Modern React development patterns and optimization techniques',
                'category': 'Technology',
                'content_type': 'Tutorial',
                'tags': ['react', 'javascript', 'frontend'],
                'creator': 'CodeMaster'
            },
            3: {
                'title': 'AI vs Human: Chess Championship',
                'description': 'Epic chess match between world champion and AI system',
                'category': 'Technology',
                'content_type': 'Live Stream',
                'tags': ['ai', 'chess', 'competition'],
                'creator': 'ChessTV'
            },
            
            # Entertainment Content
            4: {
                'title': 'Epic Movie Trailer Reactions',
                'description': 'Hilarious reactions to upcoming blockbuster movie trailers',
                'category': 'Entertainment',
                'content_type': 'Reaction Video',
                'tags': ['movies', 'reactions', 'comedy'],
                'creator': 'ReactionKing'
            },
            5: {
                'title': 'Behind the Scenes: Marvel Studios',
                'description': 'Exclusive look at how Marvel creates their superhero magic',
                'category': 'Entertainment',
                'content_type': 'Documentary',
                'tags': ['marvel', 'movies', 'bts'],
                'creator': 'MovieInsider'
            },
            6: {
                'title': 'Stand-up Comedy Special',
                'description': 'Laugh-out-loud comedy special from rising star comedian',
                'category': 'Entertainment',
                'content_type': 'Comedy Show',
                'tags': ['comedy', 'standup', 'humor'],
                'creator': 'FunnyGuy2024'
            },
            
            # Gaming Content
            7: {
                'title': 'Minecraft Epic Castle Build',
                'description': 'Time-lapse of building massive medieval castle in Minecraft',
                'category': 'Gaming',
                'content_type': 'Gaming Video',
                'tags': ['minecraft', 'building', 'timelapse'],
                'creator': 'BlockBuilder'
            },
            8: {
                'title': 'Pro Gaming Tournament Highlights',
                'description': 'Best moments from the world championship esports tournament',
                'category': 'Gaming',
                'content_type': 'Tournament Highlights',
                'tags': ['esports', 'tournament', 'gaming'],
                'creator': 'EsportsHighlights'
            },
            9: {
                'title': 'Retro Gaming Review: Classic RPGs',
                'description': 'Nostalgic review of the best role-playing games from the 90s',
                'category': 'Gaming',
                'content_type': 'Review',
                'tags': ['retro', 'rpg', 'review'],
                'creator': 'RetroGamerX'
            },
            
            # Lifestyle Content
            10: {
                'title': 'Healthy Morning Routine Ideas',
                'description': 'Transform your mornings with these energizing wellness tips',
                'category': 'Lifestyle',
                'content_type': 'Wellness Guide',
                'tags': ['health', 'morning', 'wellness'],
                'creator': 'HealthyLiving'
            },
            11: {
                'title': 'Budget Travel: Europe in 30 Days',
                'description': 'How I traveled across 15 European countries on $2000',
                'category': 'Lifestyle',
                'content_type': 'Travel Vlog',
                'tags': ['travel', 'budget', 'europe'],
                'creator': 'WanderlustSam'
            },
            12: {
                'title': 'Minimalist Home Makeover',
                'description': 'Transform cluttered space into peaceful minimalist haven',
                'category': 'Lifestyle',
                'content_type': 'Home Design',
                'tags': ['minimalism', 'home', 'design'],
                'creator': 'SimpleSpaces'
            },
            
            # Educational Content
            13: {
                'title': 'Climate Change Explained Simply',
                'description': 'Clear explanation of climate science for everyone',
                'category': 'Education',
                'content_type': 'Educational Video',
                'tags': ['climate', 'science', 'education'],
                'creator': 'ScienceSimple'
            },
            14: {
                'title': 'Ancient Rome: Rise and Fall',
                'description': 'Fascinating deep dive into Roman Empire history',
                'category': 'Education',
                'content_type': 'History Documentary',
                'tags': ['history', 'rome', 'ancient'],
                'creator': 'HistoryChannel'
            },
            15: {
                'title': 'Quantum Physics for Beginners',
                'description': 'Mind-bending concepts of quantum mechanics made accessible',
                'category': 'Education',
                'content_type': 'Science Explainer',
                'tags': ['physics', 'quantum', 'science'],
                'creator': 'PhysicsProf'
            },
            
            # Art & Creativity
            16: {
                'title': 'Digital Art Speed Painting',
                'description': 'Watch stunning fantasy artwork come to life in real-time',
                'category': 'Art & Creativity',
                'content_type': 'Art Process',
                'tags': ['art', 'digital', 'fantasy'],
                'creator': 'DigitalArtist99'
            },
            17: {
                'title': 'Photography Tips: Golden Hour',
                'description': 'Master the art of shooting during golden hour lighting',
                'category': 'Art & Creativity',
                'content_type': 'Photography Tutorial',
                'tags': ['photography', 'lighting', 'tips'],
                'creator': 'PhotoMaster'
            },
            18: {
                'title': 'DIY Craft Ideas for Beginners',
                'description': 'Easy and creative craft projects using everyday materials',
                'category': 'Art & Creativity',
                'content_type': 'DIY Tutorial',
                'tags': ['diy', 'crafts', 'creative'],
                'creator': 'CraftyCrafter'
            },
            
            # Food & Cooking
            19: {
                'title': 'Gordon Ramsay Cooking Masterclass',
                'description': 'Learn professional cooking techniques from the master chef',
                'category': 'Food & Cooking',
                'content_type': 'Cooking Tutorial',
                'tags': ['cooking', 'chef', 'masterclass'],
                'creator': 'ChefRamsay'
            },
            20: {
                'title': 'Street Food Around the World',
                'description': 'Exploring the most delicious street food from different cultures',
                'category': 'Food & Cooking',
                'content_type': 'Food Documentary',
                'tags': ['street-food', 'culture', 'travel'],
                'creator': 'FoodExplorer'
            },
            
            # Music Content
            21: {
                'title': 'Acoustic Guitar Covers Compilation',
                'description': 'Beautiful acoustic versions of popular songs',
                'category': 'Music',
                'content_type': 'Music Performance',
                'tags': ['guitar', 'acoustic', 'covers'],
                'creator': 'AcousticSoul'
            },
            22: {
                'title': 'Music Production in 10 Minutes',
                'description': 'Creating a beat from scratch using modern DAW software',
                'category': 'Music',
                'content_type': 'Music Tutorial',
                'tags': ['production', 'beats', 'daw'],
                'creator': 'BeatMaker'
            },
            
            # Sports & Fitness
            23: {
                'title': '30-Minute HIIT Workout',
                'description': 'High-intensity interval training for maximum fat burn',
                'category': 'Sports & Fitness',
                'content_type': 'Workout Video',
                'tags': ['hiit', 'fitness', 'workout'],
                'creator': 'FitTrainer'
            },
            24: {
                'title': 'NBA Greatest Dunks Compilation',
                'description': 'Mind-blowing dunks from basketball legends',
                'category': 'Sports & Fitness',
                'content_type': 'Sports Highlights',
                'tags': ['nba', 'basketball', 'dunks'],
                'creator': 'SportsCenter'
            },
            
            # News & Current Events
            25: {
                'title': 'Tech News Weekly Roundup',
                'description': 'Latest developments in technology and innovation',
                'category': 'News & Current Events',
                'content_type': 'News Summary',
                'tags': ['tech', 'news', 'innovation'],
                'creator': 'TechNewsToday'
            }
        }
        
        # Add more items programmatically for larger database
        self._generate_additional_content()
    
    def _generate_additional_content(self):
        """Generate additional content items to reach 100 items"""
        templates = [
            {
                'category': 'Technology',
                'templates': [
                    'Advanced {tech} Development Guide',
                    '{tech} vs {alt_tech}: Complete Comparison',
                    'Building {project} with {tech}',
                    '{tech} Performance Optimization Tips'
                ],
                'techs': ['JavaScript', 'Python', 'React', 'Node.js', 'Django', 'Vue.js', 'Angular', 'Docker'],
                'projects': ['E-commerce Platform', 'Social Media App', 'Portfolio Website', 'Blog System']
            },
            {
                'category': 'Entertainment',
                'templates': [
                    '{genre} Movie Reviews and Analysis',
                    'Top 10 {category} of 2024',
                    '{celebrity} Interview Highlights',
                    'Behind the Scenes: {show}'
                ],
                'genres': ['Sci-Fi', 'Horror', 'Comedy', 'Drama', 'Action'],
                'categories': ['TV Shows', 'Movies', 'Documentaries', 'Web Series'],
                'shows': ['Popular Netflix Series', 'Award-Winning Drama', 'Viral Comedy Show']
            },
            {
                'category': 'Gaming',
                'templates': [
                    '{game} Complete Walkthrough',
                    'Best {genre} Games of 2024',
                    '{game} Tips and Tricks Guide',
                    'Gaming Setup Tour: {budget} Budget'
                ],
                'games': ['Cyberpunk 2077', 'Elden Ring', 'Valorant', 'Fortnite', 'Among Us'],
                'genres': ['RPG', 'FPS', 'Strategy', 'Indie', 'Racing'],
                'budgets': ['Budget', 'Mid-Range', 'High-End']
            }
        ]
        
        item_id = 26
        for category_data in templates:
            for i in range(25):  # Generate 25 items per category
                if item_id > 100:
                    break
                    
                template = random.choice(category_data['templates'])
                
                # Fill template with random choices
                title = template
                for key, options in category_data.items():
                    if key != 'templates' and key != 'category':
                        if f'{{{key[:-1]}}}' in title:  # Remove 's' from key name
                            title = title.replace(f'{{{key[:-1]}}}', random.choice(options))
                
                # Clean up any remaining placeholders
                title = title.replace('{alt_tech}', random.choice(category_data.get('techs', ['Alternative Technology'])))
                
                self.content_items[item_id] = {
                    'title': title,
                    'description': f'Engaging content about {title.lower()}',
                    'category': category_data['category'],
                    'content_type': random.choice(['Video', 'Tutorial', 'Guide', 'Review']),
                    'tags': ['popular', 'trending', 'quality'],
                    'creator': f'Creator{item_id}'
                }
                item_id += 1
    
    def get_content(self, item_id):
        """Get content information for a specific item ID"""
        return self.content_items.get(item_id, {
            'title': f'Content Item #{item_id}',
            'description': f'Interesting content item number {item_id}',
            'category': 'General',
            'content_type': 'Content',
            'tags': ['general'],
            'creator': 'Anonymous'
        })
    
    def get_all_content(self):
        """Get all content items"""
        return self.content_items
    
    def get_content_by_category(self, category):
        """Get all content items in a specific category"""
        return {k: v for k, v in self.content_items.items() if v['category'] == category}
    
    def search_content(self, query):
        """Search content by title, description, or tags"""
        query = query.lower()
        results = {}
        for item_id, content in self.content_items.items():
            if (query in content['title'].lower() or 
                query in content['description'].lower() or 
                any(query in tag.lower() for tag in content['tags'])):
                results[item_id] = content
        return results 